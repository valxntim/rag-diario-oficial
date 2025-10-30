import sys
import os
import json
import re
import time
import concurrent.futures as cf
from tqdm import tqdm
from collections import defaultdict
import locale # Para lidar com formatação de números

# Tenta definir o locale para Português do Brasil para parsing
try:
    locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
except locale.Error:
    print("Aviso: Locale 'pt_BR.UTF-8' não encontrado. Usando locale padrão.")

# --- CONFIGURAÇÕES ---
MAX_WORKERS = 2
PER_TASK_TIMEOUT = 180
MAX_ATTEMPTS = 2
TARGET_TOTAL_CASES = 3033 
BATCH_SIZE = 25

# --- FUNÇÃO DE EXTRAÇÃO DE VALOR (v2 - Retorna String Normalizada) ---
def extract_monetary_value_v2(text: str) -> str or None:
    """
    Extrai o primeiro valor monetário encontrado e retorna como string normalizada
    no formato 'XXXX.YY' (ponto como decimal), ou None se não encontrar.
    """
    if not isinstance(text, str):
        return None

    # Patterns aprimorados, priorizando R$ e formatos claros
    patterns = [
        # Formatos claros com R$ (captura só o número)
        r'r\$\s*(\d{1,3}(?:\.\d{3})*,\d{2})', # R$ 1.234,56
        r'r\$\s*(\d+,\d{2})',                 # R$ 1234,56
        r'r\$\s*(\d{1,3}(?:,\d{3})*\.\d{2})', # R$ 1,234.56 (formato US) - Menos comum
        r'r\$\s*(\d+\.\d{2})',                 # R$ 1234.56 (formato US) - Menos comum
        r'r\$\s*(\d+)',                       # R$ 1234 (inteiro)

        # Formatos sem R$, mas com "reais" (captura só o número)
        r'(\d{1,3}(?:\.\d{3})*,\d{2})\s*reais', # 1.234,56 reais
        r'(\d+,\d{2})\s*reais',                 # 1234,56 reais

        # Busca por valor/total perto do número
        r'(?:valor|total).{0,30}?r\$\s*([\d\.,]+)', # valor/total ... R$ 1.234,56
        r'(?:valor|total).{0,30}?(\d{1,3}(?:\.\d{3})*,\d{2})', # valor/total ... 1.234,56

        # Busca genérica como último recurso
        r'(\d{1,3}(?:\.\d{3})*,\d{2})', # 1.234,56
        r'(\d+,\d{2})'                  # 1234,56
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                # Prioriza grupo de captura se existir
                value_str = match.group(1) if match.groups() else match.group(0)
                
                # Limpeza pesada: remove R$, espaços, pontos de milhar
                cleaned_str = re.sub(r'[r$\s\.]', '', value_str, flags=re.IGNORECASE)
                # Troca vírgula decimal por ponto
                normalized_str = cleaned_str.replace(',', '.')

                # Tenta converter para float para validar, mas retorna a STRING normalizada
                _ = float(normalized_str) # Lança erro se não for número válido
                
                # Adiciona .00 se for inteiro (raro, mas pode acontecer)
                if '.' not in normalized_str:
                     normalized_str += ".00"
                # Garante duas casas decimais (caso tenha uma só, ex: ,5 -> .50)
                elif len(normalized_str.split('.')[-1]) == 1:
                     normalized_str += "0"

                return normalized_str 
            except (ValueError, TypeError, IndexError):
                continue # Se a conversão falhar, tenta o próximo pattern

    return None # Se nenhum pattern funcionou

# --- Funções load_evaluation_data, load_existing_results_final, determine_work_final (sem alterações) ---
def load_evaluation_data(file_path: str):
    data = []
    if not os.path.exists(file_path):
        print(f"❌ Dataset não encontrado: {file_path}")
        return data
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
    processed_successfully = {}
    processed_failed = {}
    failed_retryable = {}
    if not os.path.exists(file_path):
        print(f"📁 Arquivo de resultados não encontrado '{file_path}', começando do zero")
        return processed_successfully, processed_failed, failed_retryable
    
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                result = json.loads(line.strip())
                pergunta_id = result.get("id_versao_pergunta")
                if not pergunta_id: continue
                count += 1
                
                acerto = result.get("acerto", False)
                resposta_gerada = result.get("resposta_gerada", "")
                
                if acerto:
                    processed_successfully[pergunta_id] = result
                elif any(err in resposta_gerada for err in ["ERRO:", "Timeout", "Exception", "vazio"]):
                    failed_retryable[pergunta_id] = result 
                else:
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
    work_items = []
    already_done_ids = set(processed_successfully.keys()) | set(processed_failed.keys())
    retry_ids = set(failed_retryable.keys())
    
    for item in evaluation_data:
        pergunta_id = item.get("id_versao_pergunta")
        if not pergunta_id: continue
        
        if pergunta_id in already_done_ids:
            continue
        elif pergunta_id in retry_ids:
            work_items.append(item)
        else:
             work_items.append(item)
             
    print(f"📊 Determinação de trabalho: {len(already_done_ids)} já concluídos (sem retentativa), {len(work_items)} para rodar/retentar")
    return work_items, already_done_ids

# --- Função build_system (sem alterações) ---
def build_system(vector_store, chunk_size, k, vizinhos, pdf_directory):
    from src.llm_interface import get_llm
    from src.rag_chain_builder import build_rag_chain_fixed
    
    llm = get_llm() 
    use_neighbors = (vizinhos == 1)
    neighbors_count = 1 if use_neighbors else 0
    
    qa_chain = build_rag_chain_fixed(
        llm=llm, vector_store=vector_store, pdf_directory=pdf_directory,
        chunk_size=chunk_size, chunk_overlap=0, use_neighbor_retriever=use_neighbors,
        k=k, neighbors=neighbors_count, force_reload=False 
    )
    
    if not qa_chain: raise RuntimeError("Falha ao inicializar RAG")
    print("✅ Sistema RAG inicializado")
    return qa_chain


# --- FUNÇÃO DE AVALIAÇÃO CORRIGIDA (v2) ---
def evaluate_one_final(qa_chain, item):
    pergunta_id = item.get("id_versao_pergunta")
    question = item.get("pergunta") 
    ground_truth_answer_str = item.get("resposta") 
    pdf = item.get("pdf", "")
    extrato = item.get("extrato")
    
    # Validação de Entrada
    if not question or not ground_truth_answer_str:
        return pergunta_id, {
            "id_versao_pergunta": pergunta_id, "pergunta": question or "", 
            "resposta_esperada": ground_truth_answer_str or "", "resposta_gerada": "ERRO: Dados de entrada inválidos (pergunta ou resposta faltando)",
            "acerto": False, "pdf": pdf, "extrato": extrato, "contextos_recuperados": [],
            "diferenca_valor": "Dados inválidos", "contextos_count": 0, "sistema": "final_processor",
            "timestamp": time.time()
        }
    
    # Execução do RAG
    try:
        rag_result = qa_chain.invoke({"query": question})
        if isinstance(rag_result, dict):
            generated_answer = (rag_result.get("result", "") or rag_result.get("answer", "") or str(rag_result))
            retrieved_contexts_docs = rag_result.get("source_documents", [])
        else:
            generated_answer = str(rag_result or "")
            retrieved_contexts_docs = []

        if not generated_answer.strip():
            generated_answer = "ERRO: LLM retornou vazio"
        
        retrieved_contexts_text = []
        for doc in retrieved_contexts_docs:
             if hasattr(doc, "page_content"): retrieved_contexts_text.append(doc.page_content)
             elif isinstance(doc, dict) and "page_content" in doc: retrieved_contexts_text.append(doc["page_content"])
             else: retrieved_contexts_text.append(str(doc))
             
    except Exception as e:
        return pergunta_id, {
            "id_versao_pergunta": pergunta_id, "pergunta": question,
            "resposta_esperada": ground_truth_answer_str, "resposta_gerada": f"ERRO CRÍTICO NA EXECUÇÃO: {str(e)}",
            "acerto": False, "pdf": pdf, "extrato": extrato, "contextos_recuperados": [],
            "diferenca_valor": None, "contextos_count": 0, "sistema": "final_processor",
            "timestamp": time.time()
        }
    
    # --- LÓGICA DE AVALIAÇÃO CORRIGIDA (v2 - Comparação de Strings Normalizadas) ---
    is_correct = False
    diff_valor = "N/A" 
    
    # 1. Extrai a STRING normalizada esperada (ex: "178719.06")
    expected_value_str = extract_monetary_value_v2(ground_truth_answer_str)
    
    # 2. Extrai a STRING normalizada gerada (ex: "178719.06" ou None)
    generated_value_str = extract_monetary_value_v2(generated_answer)
    
    # 3. COMPARAÇÃO PRIMÁRIA: Baseada nas STRINGS normalizadas
    if expected_value_str is not None:
        if generated_value_str is not None:
            # Ambos extraídos: Compara as STRINGS
            is_correct = (expected_value_str == generated_value_str)
            if is_correct:
                diff_valor = "exact_value_match"
            else:
                diff_valor = f"value_mismatch (expected={expected_value_str}, generated={generated_value_str})"
        else:
            # Esperado tinha valor, Gerado não (ou não foi extraível)
            is_correct = False
            diff_valor = f"value_not_extracted_from_generated (expected={expected_value_str}, raw_generated='{generated_answer[:50]}...')"
    else:
         # Resposta esperada não continha um valor monetário extraível.
         is_correct = False # Não podemos validar automaticamente
         diff_valor = "expected_value_not_extractable"

    # --- FIM DA LÓGICA DE AVALIAÇÃO ---

    return pergunta_id, {
        "id_versao_pergunta": pergunta_id, "pergunta": question,
        "resposta_esperada": ground_truth_answer_str,
        "resposta_gerada": generated_answer.strip().replace('\n', ' '),
        "acerto": is_correct, # <-- Reflete a comparação de strings
        "pdf": pdf, "extrato": extrato,
        "contextos_recuperados": retrieved_contexts_text,
        "diferenca_valor": diff_valor, # Log da comparação
        "contextos_count": len(retrieved_contexts_text),
        "sistema": "final_processor", "timestamp": time.time()
    }

# --- Função retry_empty_contexts (sem alterações significativas) ---
def retry_empty_contexts(output_jsonl, qa_chain):
    results = []
    try:
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): results.append(json.loads(line))
    except FileNotFoundError:
        print(f"⚠️ Aviso: Arquivo {output_jsonl} não encontrado para retentativa.")
        return 

    to_retry = [x for x in results if x.get('contextos_count', 0) == 0 and not x.get('resposta_gerada','').startswith("ERRO:")]
    
    if not to_retry:
        print("✅ Nenhuma retentativa necessária para contextos vazios.")
        return
        
    print(f"\n🔁 Retrying {len(to_retry)} cases with empty contextos...")
    updated = {x['id_versao_pergunta']: x for x in results} 
    
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = { executor.submit(evaluate_one_final, qa_chain, item): item['id_versao_pergunta'] for item in to_retry }
        for future in tqdm(cf.as_completed(futures), total=len(to_retry), desc="Retentando contextos vazios"):
            try:
                pid, result = future.result()
                updated[pid] = result 
            except Exception as e:
                 pid_original = futures[future]
                 print(f"❌ Erro na retentativa de {pid_original}: {e}")
                 if pid_original in updated: updated[pid_original]["resposta_gerada"] = f"ERRO NA RETENTATIVA: {e}"

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for obj in updated.values():
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    print("✅ Retentativa dos contextos vazios concluída.")


# --- Função run_final_evaluation (sem alterações significativas na lógica principal) ---
def run_final_evaluation(input_file, output_file, vector_store, chunk_size, k, vizinhos, pdf_directory):
    print(f"🚀 Iniciando avaliação: {input_file} -> {output_file}")
    evaluation_data = load_evaluation_data(input_file)
    if not evaluation_data: return
    
    qa_chain = build_system(vector_store, chunk_size, k, vizinhos, pdf_directory)
    
    if os.path.exists(output_file):
        import shutil
        backup_file = output_file + ".bak"
        try:
            shutil.copy2(output_file, backup_file)
            print(f"💾 Backup criado: {backup_file}")
        except Exception as e:
            print(f"⚠️ Aviso: Falha ao criar backup. {e}")
            
    processed_successfully, processed_failed, failed_retryable = load_existing_results_final(output_file)
    total_processed_definitivo = len(processed_successfully) + len(processed_failed)
    run_count = 0

    work_items, already_done_ids = determine_work_final(evaluation_data, processed_successfully, processed_failed, failed_retryable)
    
    pbar_global = tqdm(total=TARGET_TOTAL_CASES, initial=total_processed_definitivo, desc="Progresso Geral")
    processed_in_this_run = set() 

    while work_items:
        run_count += 1
        print(f"\n🔄 === RUN {run_count} ===")
        
        batch = work_items[:BATCH_SIZE]
        work_items = work_items[BATCH_SIZE:] 
        
        batch_to_run = [item for item in batch if item.get("id_versao_pergunta") not in processed_in_this_run]
        if not batch_to_run: continue 
        
        results_batch = {}
        
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
                             "id_versao_pergunta": pid_original, "pergunta": "", "resposta_esperada": "",
                             "resposta_gerada": f"ERRO FATAL NO WORKER: {e}", "acerto": False,
                         }

        for pid, result in results_batch.items():
            if result.get("acerto", False): processed_successfully[pid] = result
            elif any(err in result.get("resposta_gerada","") for err in ["ERRO:", "Timeout", "Exception", "vazio"]): failed_retryable[pid] = result
            else: processed_failed[pid] = result
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                all_results = list(processed_successfully.values()) + list(processed_failed.values()) + list(failed_retryable.values())
                for obj in all_results:
                    f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"❌ ERRO ao salvar resultados no Run {run_count}: {e}")

        total_processed_agora = len(processed_successfully) + len(processed_failed)
        pbar_global.n = total_processed_agora
        pbar_global.refresh()
        print(f"📈 Run {run_count} concluído. Total definitivo: {total_processed_agora}")

        if total_processed_agora >= TARGET_TOTAL_CASES:
            print(f"🎊 Meta atingida! {total_processed_agora} casos processados.")
            break
        if not work_items and failed_retryable: 
             print("⏳ Forçando novo ciclo para retentar falhas técnicas...")
             processed_successfully, processed_failed, failed_retryable = load_existing_results_final(output_file)
             work_items, _ = determine_work_final(evaluation_data, processed_successfully, processed_failed, failed_retryable)

    pbar_global.close()
    
    print("\n--- Iniciando Retentativa Final para Contextos Vazios ---")
    retry_empty_contexts(output_file, qa_chain)
    
    print("\n🏁 Avaliação Final Concluída.")

# --- Bloco Principal ---
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