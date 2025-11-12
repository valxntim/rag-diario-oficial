"""
🧪 AVALIAÇÃO FINAL DO SISTEMA RAG
==================================
Script que executa o pipeline RAG completo contra um dataset de perguntas/respostas.
Avalia a qualidade das respostas extraindo e comparando valores monetários.
Implementa retry automático, cache e salva resultados em JSONL.
"""

import sys
import os
import json
import re
import time
import concurrent.futures as cf
from tqdm import tqdm
from collections import defaultdict
import locale  # Para lidar com formatação de números em PT-BR


# ========== CONFIGURAÇÃO DE LOCALE ==========
# Define o locale para Português do Brasil para parsing correto de números
try:
    locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
except locale.Error:
    print("Aviso: Locale 'pt_BR.UTF-8' não encontrado. Usando locale padrão.")


# ========== CONFIGURAÇÕES DE EXECUÇÃO ==========
# Parâmetros que controlam como a avaliação roda

MAX_WORKERS = 2                    # Número de threads paralelas para processar perguntas
PER_TASK_TIMEOUT = 180             # Timeout em segundos por pergunta (evita travar)
MAX_ATTEMPTS = 2                   # Número máximo de tentativas se falhar
TARGET_TOTAL_CASES = 261           # Meta de casos a processar
BATCH_SIZE = 25                    # Quantidade de perguntas por lote


# ========== FUNÇÃO DE EXTRAÇÃO DE VALOR MONETÁRIO ==========
def extract_monetary_value_v2(text: str) -> str or None:
    """
    Extrai o PRIMEIRO valor monetário encontrado no texto e retorna como string normalizada.
    
    Estratégia:
    1. Tenta múltiplos padrões regex (R$, reais, valor/total, genérico)
    2. Limpa o valor (remove R$, espaços, pontos de milhar)
    3. Normaliza para formato "XXXX.YY" (ponto como decimal)
    4. Valida conversão para float
    5. Garante 2 casas decimais
    
    Argumentos:
        text: Texto onde procurar valor (ex: resposta do LLM)
    
    Retorna:
        String normalizada (ex: "178719.06") ou None se não encontrar
    
    Exemplo:
        extract_monetary_value_v2("Valor encontrado: R$ 1.234,56")
        → "1234.56"
    """
    # Validação básica: texto deve ser string
    if not isinstance(text, str):
        return None

    # ========== PADRÕES REGEX (em ordem de prioridade) ==========
    # Quanto mais específico, mais alta a prioridade (aparece antes)
    patterns = [
        # 1. Formatos CLAROS com R$ (padrão PT-BR: ponto de milhar, vírgula decimal)
        r'r\$\s*(\d{1,3}(?:\.\d{3})*,\d{2})',  # Ex: R$ 1.234,56
        r'r\$\s*(\d+,\d{2})',                  # Ex: R$ 1234,56
        r'r\$\s*(\d{1,3}(?:,\d{3})*\.\d{2})',  # Ex: R$ 1,234.56 (formato US) - Menos comum
        r'r\$\s*(\d+\.\d{2})',                 # Ex: R$ 1234.56 (formato US) - Menos comum
        r'r\$\s*(\d+)',                        # Ex: R$ 1234 (inteiro)

        # 2. Formatos sem R$, mas com "reais" (palavra chave)
        r'(\d{1,3}(?:\.\d{3})*,\d{2})\s*reais',  # Ex: 1.234,56 reais
        r'(\d+,\d{2})\s*reais',                  # Ex: 1234,56 reais

        # 3. Busca por contexto (valor/total perto do número)
        r'(?:valor|total).{0,30}?r\$\s*([\d\.,]+)',  # valor/total ... R$ 1.234,56
        r'(?:valor|total).{0,30}?(\d{1,3}(?:\.\d{3})*,\d{2})',  # valor/total ... 1.234,56

        # 4. Padrões genéricos (último recurso - menos preciso)
        r'(\d{1,3}(?:\.\d{3})*,\d{2})',  # 1.234,56
        r'(\d+,\d{2})'                   # 1234,56
    ]

    # ========== BUSCA PELOS PADRÕES ==========
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                # Extrai o valor encontrado (prioriza grupo de captura se houver)
                value_str = match.group(1) if match.groups() else match.group(0)
                
                # Limpeza pesada: remove caracteres não-numéricos (R$, espaços, pontos de milhar)
                cleaned_str = re.sub(r'[r$\s\.]', '', value_str, flags=re.IGNORECASE)
                
                # Troca vírgula (decimal português) por ponto (decimal padrão interno)
                normalized_str = cleaned_str.replace(',', '.')

                # Validação: tenta converter para float (lança erro se inválido)
                _ = float(normalized_str)
                
                # ========== NORMALIZAÇÃO FINAL ==========
                # Adiciona .00 se for inteiro (raro, mas pode acontecer)
                if '.' not in normalized_str:
                    normalized_str += ".00"
                # Garante exatamente 2 casas decimais (ex: .5 → .50)
                elif len(normalized_str.split('.')[-1]) == 1:
                    normalized_str += "0"

                return normalized_str  # Retorna string normalizada
                
            except (ValueError, TypeError, IndexError):
                # Se a conversão falhar, tenta o próximo padrão
                continue

    return None  # Se NENHUM padrão funcionou


# ========== FUNÇÕES DE CARREGAMENTO E PERSISTÊNCIA ==========

def load_evaluation_data(file_path: str):
    """
    Carrega dataset de avaliação em formato JSONL (1 JSON por linha).
    
    Cada linha deve ter: id_versao_pergunta, pergunta, resposta, pdf, extrato
    
    Argumentos:
        file_path: Caminho para o arquivo JSONL
    
    Retorna:
        Lista de dicts com as perguntas/respostas de referência
    """
    data = []
    if not os.path.exists(file_path):
        print(f"❌ Dataset não encontrado: {file_path}")
        return data
    
    # Lê arquivo linha por linha (formato JSONL)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"⚠️ Aviso: Linha inválida pulada em {file_path}")
                continue
    
    print(f"📊 Dataset carregado: {len(data)} casos de {file_path}")
    return data


def load_existing_results_final(file_path: str):
    """
    Carrega resultados anteriores para retomar de onde parou.
    
    Classifica resultados em 3 categorias:
    - Sucessos: acerto = True (valor correto encontrado)
    - Falhas de conteúdo: acerto = False (mas nenhum erro técnico)
    - Falhas técnicas: contém "ERRO:", "Timeout", etc (retentáveis)
    
    Argumentos:
        file_path: Caminho do arquivo de resultados (JSONL)
    
    Retorna:
        Tupla de 3 dicts: (processed_successfully, processed_failed, failed_retryable)
    """
    processed_successfully = {}
    processed_failed = {}
    failed_retryable = {}
    
    if not os.path.exists(file_path):
        print(f"📁 Arquivo de resultados não encontrado '{file_path}', começando do zero")
        return processed_successfully, processed_failed, failed_retryable
    
    count = 0
    # Lê arquivo JSONL linha por linha
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): 
                continue
            try:
                result = json.loads(line.strip())
                pergunta_id = result.get("id_versao_pergunta")
                if not pergunta_id: 
                    continue
                count += 1
                
                acerto = result.get("acerto", False)
                resposta_gerada = result.get("resposta_gerada", "")
                
                # Classifica em uma das 3 categorias
                if acerto:
                    processed_successfully[pergunta_id] = result
                elif any(err in resposta_gerada for err in ["ERRO:", "Timeout", "Exception", "vazio"]):
                    # Erro técnico = retentável
                    failed_retryable[pergunta_id] = result 
                else:
                    # Erro de conteúdo = não retenta
                    processed_failed[pergunta_id] = result 
            except json.JSONDecodeError:
                print(f"⚠️ Aviso: Linha inválida pulada em {file_path}")
                continue
                
    print(f"💾 Resultados anteriores carregados de '{file_path}':")
    print(f"   - Sucessos: {len(processed_successfully)}")
    print(f"   - Falhas (conteúdo): {len(processed_failed)}")
    print(f"   - Falhas (técnicas/retentáveis): {len(failed_retryable)}")
    print(f"   - Total lido: {count}")
    
    return processed_successfully, processed_failed, failed_retryable


def determine_work_final(evaluation_data, processed_successfully, processed_failed, failed_retryable):
    """
    Determina quais perguntas ainda precisam ser processadas.
    
    Lógica:
    - Se pergunta já foi bem-sucedida OU falhou (conteúdo): PULA
    - Se pergunta falhou com erro técnico: RETENTA
    - Se pergunta é nova: PROCESSA
    
    Argumentos:
        evaluation_data: Dataset completo de perguntas
        processed_successfully: Perguntas já resolvidas
        processed_failed: Perguntas que falharam (conteúdo)
        failed_retryable: Perguntas que falharam (técnico)
    
    Retorna:
        Tupla: (work_items, already_done_ids) - perguntas a processar e IDs já processadas
    """
    work_items = []
    already_done_ids = set(processed_successfully.keys()) | set(processed_failed.keys())
    retry_ids = set(failed_retryable.keys())
    
    for item in evaluation_data:
        pergunta_id = item.get("id_versao_pergunta")
        if not pergunta_id: 
            continue
        
        if pergunta_id in already_done_ids:
            # Já foi processada e finalizada: PULA
            continue
        elif pergunta_id in retry_ids:
            # Falhou tecnicamente: RETENTA
            work_items.append(item)
        else:
            # Nova: PROCESSA
            work_items.append(item)
             
    print(f"📊 Determinação de trabalho: {len(already_done_ids)} já concluídos (sem retentativa), {len(work_items)} para rodar/retentar")
    return work_items, already_done_ids


# ========== FUNÇÃO PARA MONTAR O SISTEMA RAG ==========

def build_system(vector_store, chunk_size, k, vizinhos, pdf_directory):
    """
    Inicializa o sistema RAG (LLM + RAG Chain).
    
    Argumentos:
        vector_store: Índice FAISS já carregado
        chunk_size: Tamanho dos chunks
        k: Número de chunks a recuperar
        vizinhos: 1 = usar neighbor retriever, 0 = retriever básico
        pdf_directory: Pasta dos PDFs (necessária para neighbor retriever)
    
    Retorna:
        qa_chain: RetrievalQA pronta para .invoke({"query": "..."})
    """
    from src.llm_interface import get_llm
    from src.rag_chain_builder import build_rag_chain_fixed
    
    # Carrega o LLM (Ollama)
    llm = get_llm() 
    
    # Interpreta flag de vizinhos
    use_neighbors = (vizinhos == 1)
    neighbors_count = 1 if use_neighbors else 0
    
    # Monta a cadeia RAG
    qa_chain = build_rag_chain_fixed(
        llm=llm, 
        vector_store=vector_store, 
        pdf_directory=pdf_directory,
        chunk_size=chunk_size, 
        chunk_overlap=0, 
        use_neighbor_retriever=use_neighbors,
        k=k, 
        neighbors=neighbors_count, 
        force_reload=False 
    )
    
    if not qa_chain: 
        raise RuntimeError("Falha ao inicializar RAG")
    print("✅ Sistema RAG inicializado")
    return qa_chain


# ========== FUNÇÃO PRINCIPAL DE AVALIAÇÃO ==========

def evaluate_one_final(qa_chain, item):
    """
    Avalia UMA pergunta contra o sistema RAG.
    
    Processo:
    1. Extrai pergunta/resposta esperada do item
    2. Executa RAG chain (busca + LLM)
    3. Compara resposta gerada vs esperada (extrai valores monetários)
    4. Retorna resultado com metadados
    
    Argumentos:
        qa_chain: RetrievalQA chain (já inicializada)
        item: Dict com pergunta, resposta esperada, etc
    
    Retorna:
        Tupla: (pergunta_id, resultado_dict)
    """
    # Extrai dados do item
    pergunta_id = item.get("id_versao_pergunta")
    question = item.get("pergunta") 
    ground_truth_answer_str = item.get("resposta") 
    pdf = item.get("pdf", "")
    extrato = item.get("extrato")
    
    # ========== VALIDAÇÃO DE ENTRADA ==========
    if not question or not ground_truth_answer_str:
        return pergunta_id, {
            "id_versao_pergunta": pergunta_id, 
            "pergunta": question or "", 
            "resposta_esperada": ground_truth_answer_str or "", 
            "resposta_gerada": "ERRO: Dados de entrada inválidos (pergunta ou resposta faltando)",
            "acerto": False, 
            "pdf": pdf, 
            "extrato": extrato, 
            "contextos_recuperados": [],
            "diferenca_valor": "Dados inválidos", 
            "contextos_count": 0, 
            "sistema": "final_processor",
            "timestamp": time.time()
        }
    
    # ========== EXECUÇÃO DO RAG ==========
    try:
        # Invoca a chain RAG com a pergunta
        rag_result = qa_chain.invoke({"query": question})
        
        # Extrai resposta e documentos recuperados (estrutura pode variar)
        if isinstance(rag_result, dict):
            generated_answer = (rag_result.get("result", "") or rag_result.get("answer", "") or str(rag_result))
            retrieved_contexts_docs = rag_result.get("source_documents", [])
        else:
            generated_answer = str(rag_result or "")
            retrieved_contexts_docs = []

        if not generated_answer.strip():
            generated_answer = "ERRO: LLM retornou vazio"
        
        # Extrai conteúdo dos documentos recuperados
        retrieved_contexts_text = []
        for doc in retrieved_contexts_docs:
            if hasattr(doc, "page_content"): 
                retrieved_contexts_text.append(doc.page_content)
            elif isinstance(doc, dict) and "page_content" in doc: 
                retrieved_contexts_text.append(doc["page_content"])
            else: 
                retrieved_contexts_text.append(str(doc))
             
    except Exception as e:
        # Erro durante execução do RAG
        return pergunta_id, {
            "id_versao_pergunta": pergunta_id, 
            "pergunta": question,
            "resposta_esperada": ground_truth_answer_str, 
            "resposta_gerada": f"ERRO CRÍTICO NA EXECUÇÃO: {str(e)}",
            "acerto": False, 
            "pdf": pdf, 
            "extrato": extrato, 
            "contextos_recuperados": [],
            "diferenca_valor": None, 
            "contextos_count": 0, 
            "sistema": "final_processor",
            "timestamp": time.time()
        }
    
    # ========== LÓGICA DE AVALIAÇÃO (Comparação de Strings Normalizadas) ==========
    is_correct = False
    diff_valor = "N/A" 
    
    # 1. Extrai a STRING normalizada ESPERADA (ex: "178719.06")
    expected_value_str = extract_monetary_value_v2(ground_truth_answer_str)
    
    # 2. Extrai a STRING normalizada GERADA (ex: "178719.06" ou None)
    generated_value_str = extract_monetary_value_v2(generated_answer)
    
    # 3. COMPARAÇÃO: Baseada nas STRINGS normalizadas
    if expected_value_str is not None:
        if generated_value_str is not None:
            # Ambos extraídos: Compara as strings diretamente
            is_correct = (expected_value_str == generated_value_str)
            if is_correct:
                diff_valor = "exact_value_match"
            else:
                diff_valor = f"value_mismatch (expected={expected_value_str}, generated={generated_value_str})"
        else:
            # Esperado tinha valor, Gerado NÃO
            is_correct = False
            diff_valor = f"value_not_extracted_from_generated (expected={expected_value_str}, raw_generated='{generated_answer[:50]}...')"
    else:
        # Resposta esperada não continha valor monetário extraível
        is_correct = False
        diff_valor = "expected_value_not_extractable"

    # ========== RETORNO DOS RESULTADOS ==========
    return pergunta_id, {
        "id_versao_pergunta": pergunta_id, 
        "pergunta": question,
        "resposta_esperada": ground_truth_answer_str,
        "resposta_gerada": generated_answer.strip().replace('\n', ' '),
        "acerto": is_correct,
        "pdf": pdf, 
        "extrato": extrato,
        "contextos_recuperados": retrieved_contexts_text,
        "diferenca_valor": diff_valor,
        "contextos_count": len(retrieved_contexts_text),
        "sistema": "final_processor", 
        "timestamp": time.time()
    }


# ========== FUNÇÃO DE RETENTATIVA ==========

def retry_empty_contexts(output_jsonl, qa_chain):
    """
    Retenta avaliar perguntas que retornaram SEM CONTEXTO (0 documentos recuperados).
    
    Lógica: Se a busca não retornou nenhum documento, pode ser falha temporária.
    Retenta esperando melhor sorte.
    
    Argumentos:
        output_jsonl: Arquivo de resultados
        qa_chain: RAG chain para retentativa
    """
    results = []
    try:
        # Carrega todos os resultados anteriores
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): 
                    results.append(json.loads(line))
    except FileNotFoundError:
        print(f"⚠️ Aviso: Arquivo {output_jsonl} não encontrado para retentativa.")
        return 

    # Filtra apenas perguntas que retornaram 0 contextos (e não têm "ERRO:")
    to_retry = [x for x in results if x.get('contextos_count', 0) == 0 and not x.get('resposta_gerada','').startswith("ERRO:")]
    
    if not to_retry:
        print("✅ Nenhuma retentativa necessária para contextos vazios.")
        return
        
    print(f"\n🔁 Retrying {len(to_retry)} cases with empty contextos...")
    updated = {x['id_versao_pergunta']: x for x in results} 
    
    # Retenta em paralelo
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = { executor.submit(evaluate_one_final, qa_chain, item): item['id_versao_pergunta'] for item in to_retry }
        for future in tqdm(cf.as_completed(futures), total=len(to_retry), desc="Retentando contextos vazios"):
            try:
                pid, result = future.result()
                updated[pid] = result 
            except Exception as e:
                pid_original = futures[future]
                print(f"❌ Erro na retentativa de {pid_original}: {e}")
                if pid_original in updated: 
                    updated[pid_original]["resposta_gerada"] = f"ERRO NA RETENTATIVA: {e}"

    # Salva resultados atualizados
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for obj in updated.values():
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    print("✅ Retentativa dos contextos vazios concluída.")


# ========== FUNÇÃO PRINCIPAL DE AVALIAÇÃO ==========

def run_final_evaluation(input_file, output_file, vector_store, chunk_size, k, vizinhos, pdf_directory):
    """
    Executa a avaliação COMPLETA do sistema RAG contra um dataset.
    
    Pipeline:
    1. Carrega dataset de perguntas
    2. Inicializa RAG chain
    3. Processa em lotes paralelos (com retry automático)
    4. Salva resultados incrementalmente
    5. Retenta perguntas com contexto vazio
    
    Argumentos:
        input_file: Arquivo JSONL com dataset
        output_file: Arquivo JSONL onde salvar resultados
        vector_store: Índice FAISS
        chunk_size: Tamanho dos chunks
        k: Top K a recuperar
        vizinhos: 1=com neighbors, 0=sem
        pdf_directory: Pasta dos PDFs
    """
    print(f"🚀 Iniciando avaliação: {input_file} -> {output_file}")
    
    # Carrega dataset
    evaluation_data = load_evaluation_data(input_file)
    if not evaluation_data: 
        return
    
    # Inicializa RAG
    qa_chain = build_system(vector_store, chunk_size, k, vizinhos, pdf_directory)
    
    # Cria backup do arquivo de resultados (se existir)
    if os.path.exists(output_file):
        import shutil
        backup_file = output_file + ".bak"
        try:
            shutil.copy2(output_file, backup_file)
            print(f"💾 Backup criado: {backup_file}")
        except Exception as e:
            print(f"⚠️ Aviso: Falha ao criar backup. {e}")
    
    # Carrega resultados anteriores (para retomar)
    processed_successfully, processed_failed, failed_retryable = load_existing_results_final(output_file)
    total_processed_definitivo = len(processed_successfully) + len(processed_failed)
    run_count = 0

    # Determina quais perguntas ainda faltam processar
    work_items, already_done_ids = determine_work_final(evaluation_data, processed_successfully, processed_failed, failed_retryable)
    
    # Progress bar global
    pbar_global = tqdm(total=TARGET_TOTAL_CASES, initial=total_processed_definitivo, desc="Progresso Geral")
    processed_in_this_run = set() 

    # ========== LOOP PRINCIPAL ==========
    while work_items:
        run_count += 1
        print(f"\n🔄 === RUN {run_count} ===")
        
        # Divide em lotes (BATCH_SIZE perguntas por lote)
        batch = work_items[:BATCH_SIZE]
        work_items = work_items[BATCH_SIZE:] 
        
        # Filtra perguntas que já foram processadas nesta execução
        batch_to_run = [item for item in batch if item.get("id_versao_pergunta") not in processed_in_this_run]
        if not batch_to_run: 
            continue 
        
        results_batch = {}
        
        # ========== PROCESSAMENTO PARALELO ==========
        # Usa ThreadPoolExecutor para processar múltiplas perguntas em paralelo
        with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = { executor.submit(evaluate_one_final, qa_chain, item): item.get("id_versao_pergunta") for item in batch_to_run }
            for future in tqdm(cf.as_completed(futures), total=len(batch_to_run), desc=f"Processando Run {run_count}"):
                try:
                    pid, result = future.result()
                    if pid:
                        results_batch[pid] = result
                        processed_in_this_run.add(pid)
                except Exception as e:
                    pid_original = futures[future]
                    print(f"❌ Erro grave no future para {pid_original}: {e}")
                    if pid_original:
                        results_batch[pid_original] = {
                            "id_versao_pergunta": pid_original, 
                            "pergunta": "", 
                            "resposta_esperada": "",
                            "resposta_gerada": f"ERRO FATAL NO WORKER: {e}", 
                            "acerto": False,
                        }

        # ========== CLASSIFICAÇÃO DE RESULTADOS ==========
        for pid, result in results_batch.items():
            if result.get("acerto", False): 
                processed_successfully[pid] = result
            elif any(err in result.get("resposta_gerada","") for err in ["ERRO:", "Timeout", "Exception", "vazio"]): 
                failed_retryable[pid] = result
            else: 
                processed_failed[pid] = result
        
        # ========== SALVA RESULTADOS INCREMENTALMENTE ==========
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                all_results = list(processed_successfully.values()) + list(processed_failed.values()) + list(failed_retryable.values())
                for obj in all_results:
                    f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"❌ ERRO ao salvar resultados no Run {run_count}: {e}")

        # Atualiza progress bar
        total_processed_agora = len(processed_successfully) + len(processed_failed)
        pbar_global.n = total_processed_agora
        pbar_global.refresh()
        print(f"📈 Run {run_count} concluído. Total definitivo: {total_processed_agora}")

        # Verifica se atingiu meta
        if total_processed_agora >= TARGET_TOTAL_CASES:
            print(f"🎊 Meta atingida! {total_processed_agora} casos processados.")
            break
        
        # Se não há mais itens novos mas há retentáveis, força novo ciclo
        if not work_items and failed_retryable: 
            print("⏳ Forçando novo ciclo para retentar falhas técnicas...")
            processed_successfully, processed_failed, failed_retryable = load_existing_results_final(output_file)
            work_items, _ = determine_work_final(evaluation_data, processed_successfully, processed_failed, failed_retryable)

    pbar_global.close()
    
    # ========== RETENTATIVA FINAL ==========
    print("\n--- Iniciando Retentativa Final para Contextos Vazios ---")
    retry_empty_contexts(output_file, qa_chain)
    
    print("\n🏁 Avaliação Final Concluída.")


# ========== BLOCO PRINCIPAL ==========
if __name__ == "__main__":
    print("final_eval.py carregado como módulo ou script.")
    # Adicione aqui a lógica se precisar executar diretamente
    # Exemplo:
    # if len(sys.argv) == 3:
    #     input_f = sys.argv[1]
    #     output_f = sys.argv[2]
    #     # Você precisaria carregar/configurar vector_store, chunk_size etc. aqui
    #     print("Execução direta não implementada neste exemplo.")
    # else:
    #     print("Argumentos ausentes para execução direta.")