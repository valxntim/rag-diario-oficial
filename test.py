#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT DE AVALIAÇÃO RAG - VERSÃO SEM SIGNALS (CORRIGIDA)
Remove uso de signals para funcionar com ThreadPoolExecutor
Desenvolvido por: Senior Software Engineer
"""

import sys
import os
import json
import re
import time
import concurrent.futures as cf
from tqdm import tqdm
from collections import defaultdict
import threading
import functools
from typing import Optional, Any, Dict, List, Union


# --- CONFIGURAÇÕES GLOBAIS ---
MAX_WORKERS = 1  # 1 worker para evitar problemas com signals
PER_TASK_TIMEOUT = 120
MAX_ATTEMPTS = 3
TARGET_TOTAL_CASES = 294
BATCH_SIZE = 20


# --- CLASSES DE EXCEÇÃO ---
class TimeoutException(Exception):
    """Exceção específica para timeouts"""
    pass


class RetryableError(Exception):
    """Exceção para erros que podem ser tentados novamente"""
    pass


# --- TIMEOUT SEM SIGNALS (Threading-based) ---
class ThreadTimeout:
    """Implementa timeout usando threading ao invés de signals"""
    
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.timer = None
        self.timed_out = False
    
    def _timeout_handler(self):
        """Handler chamado quando timeout ocorre"""
        self.timed_out = True
    
    def __enter__(self):
        # Criar timer que será executado após timeout_seconds
        self.timer = threading.Timer(self.timeout_seconds, self._timeout_handler)
        self.timer.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cancelar timer se ainda não executou
        if self.timer:
            self.timer.cancel()
    
    def check_timeout(self):
        """Verifica se timeout ocorreu - deve ser chamado periodicamente"""
        if self.timed_out:
            raise TimeoutException(f"Operação excedeu {self.timeout_seconds} segundos")


def with_thread_timeout(timeout_seconds):
    """Decorator que implementa timeout usando threading"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = None
            exception = None
            
            def target():
                nonlocal result, exception
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    exception = e
            
            # Executar função em thread separada com timeout
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            
            # Aguardar com timeout
            thread.join(timeout=timeout_seconds)
            
            if thread.is_alive():
                # Thread ainda está executando = timeout
                raise TimeoutException(f"Função excedeu timeout de {timeout_seconds}s")
            
            if exception:
                raise exception
            
            return result
        
        return wrapper
    return decorator


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0
):
    """Decorator para retry com exponential backoff - SEM SIGNALS"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (TimeoutException, ConnectionError, RetryableError) as e:
                    last_exception = e
                    
                    # Se for o último attempt, não faz retry
                    if attempt == max_attempts - 1:
                        print(f"❌ Falhou após {max_attempts} tentativas: {str(e)}")
                        break
                    
                    # Log do retry
                    print(f"🔄 Tentativa {attempt + 1}/{max_attempts} falhou: {str(e)}")
                    print(f"⏳ Aguardando {delay:.1f}s antes da próxima tentativa...")
                    
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
                except Exception as e:
                    # Erros não retryáveis
                    print(f"❌ Erro não retryável: {str(e)}")
                    raise e
            
            # Se chegou aqui, todas as tentativas falharam
            raise last_exception
        return wrapper
    return decorator


# --- FUNÇÕES AUXILIARES ORIGINAIS ---
def extract_monetary_value_fast(text: str) -> str or None:
    """Extrai valor monetário do texto - função original mantida"""
    if not isinstance(text, str):
        return None
    patterns = [
        r'R\$\s*([\d\.,]+)',
        r'(\d{1,3}(?:\.\d{3})*,\d{2})\s*reais',
        r'valor.{0,50}?(\d{1,3}(?:\.\d{3})*,\d{2})',
        r'total.{0,50}?(\d{1,3}(?:\.\d{3})*,\d{2})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value_str = match.group(1) if len(match.groups()) > 0 else match.group(0)
                value_str = re.sub(r'^R\$\s*', '', value_str)
                if ',' in value_str and '.' in value_str:
                    if value_str.count(',') == 1 and value_str.rfind(',') > value_str.rfind('.'):
                        normalized = value_str.replace('.', '').replace(',', '.')
                    else:
                        normalized = value_str.replace(',', '').replace('.', '')
                elif ',' in value_str:
                    parts = value_str.split(',')
                    if len(parts) == 2 and len(parts[1]) <= 2:
                        normalized = value_str.replace(',', '.')
                    else:
                        normalized = value_str.replace(',', '')
                else:
                    normalized = value_str.replace('.', '')
                return normalized
            except:
                continue
    return None


def load_evaluation_data(file_path: str):
    """Carrega dados de avaliação do arquivo JSONL"""
    data = []
    if not os.path.exists(file_path):
        print(f"❌ Dataset não encontrado: {file_path}")
        return data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    print(f"📊 Dataset carregado: {len(data)} casos")
    return data


def load_existing_results_final(file_path: str):
    """Carrega resultados existentes do arquivo de saída"""
    processed_successfully = {}
    processed_failed = {}
    failed_retryable = {}
    
    if not os.path.exists(file_path):
        print(f"📁 Arquivo não encontrado, começando do zero")
        return processed_successfully, processed_failed, failed_retryable
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                result = json.loads(line.strip())
                pergunta_id = result.get("id_versao_pergunta")
                if not pergunta_id:
                    continue
                
                acerto = result.get("acerto", False)
                resposta_gerada = result.get("resposta_gerada", "")
                
                if acerto:
                    processed_successfully[pergunta_id] = result
                elif any(erro_tecnico in resposta_gerada for erro_tecnico in [
                    "ERRO: Timeout", "ERRO NA EXECUÇÃO", "ERRO CRÍTICO",
                    "Exception", "TimeoutError", "TimeoutException"
                ]):
                    failed_retryable[pergunta_id] = result
                else:
                    processed_failed[pergunta_id] = result
            except json.JSONDecodeError:
                continue
    
    return processed_successfully, processed_failed, failed_retryable


def determine_work_final(evaluation_data, processed_successfully, processed_failed, failed_retryable):
    """Determina quais itens ainda precisam ser processados"""
    work_items = []
    already_done = set()
    all_processed = set(processed_successfully.keys()) | set(processed_failed.keys())
    
    for item in evaluation_data:
        pergunta_id = item.get("id_versao_pergunta")
        if not pergunta_id:
            continue
        if pergunta_id in all_processed:
            already_done.add(pergunta_id)
        else:
            work_items.append(item)
    
    print(f"📊 Determinação de trabalho: {len(already_done)} já feitos, {len(work_items)} para rodar")
    return work_items, already_done


def build_system():
    """Constrói o sistema RAG"""
    from src.llm_interface import get_llm
    from src.vector_store_manager import get_vector_store
    from src.rag_chain_builder import build_rag_chain
    
    vector_store = get_vector_store(force_recreate=False)
    llm = get_llm()
    qa_chain = build_rag_chain(llm, vector_store)
    
    if not all([vector_store, llm, qa_chain]):
        raise RuntimeError("Falha ao inicializar RAG")
    
    print("✅ Sistema RAG inicializado")
    return qa_chain


def create_error_result(item: dict, error_message: str) -> dict:
    """Cria resultado de erro padronizado"""
    return {
        "id_versao_pergunta": item.get("id_versao_pergunta"),
        "pergunta": item.get("pergunta", ""),
        "resposta_esperada": item.get("resposta", "") or item.get("answer", ""),
        "resposta_gerada": error_message,
        "acerto": False,
        "pdf": item.get("pdf", ""),
        "extrato": item.get("extrato"),
        "contextos_recuperados": [],
        "diferenca_valor": None,
        "contextos_count": 0,
        "sistema": "final_processor_v3_no_signals",
        "timestamp": time.time()
    }


# --- FUNÇÃO PRINCIPAL SEM SIGNALS ---
@retry_with_backoff(max_attempts=MAX_ATTEMPTS, initial_delay=1.0, backoff_factor=2.0)
@with_thread_timeout(PER_TASK_TIMEOUT)
def evaluate_one_final(qa_chain, item):
    """
    VERSÃO SEM SIGNALS: Avalia um item com timeout usando threading
    *** CORRIGE O PROBLEMA DO SIGNAL EM THREADS ***
    """
    pergunta_id = item.get("id_versao_pergunta")
    question = item.get("pergunta")
    ground_truth_answer = item.get("resposta") or item.get("answer")
    pdf = item.get("pdf", "")
    extrato = item.get("extrato")
    
    if not question or not ground_truth_answer:
        return pergunta_id, create_error_result(item, "ERRO: Dados inválidos")
    
    try:
        print(f"🚀 Processando pergunta: {pergunta_id}")
        
        # *** CHAMADA DO LLM AGORA COM TIMEOUT VIA THREADING ***
        rag_result = qa_chain.invoke({"query": question})
        
        if isinstance(rag_result, dict):
            generated_answer = (
                rag_result.get("result", "") or 
                rag_result.get("answer", "") or 
                str(rag_result)
            )
        else:
            generated_answer = str(rag_result or "")
        
        if not generated_answer.strip():
            raise RetryableError("LLM retornou resposta vazia")
        
        # Extract contexts
        retrieved_contexts = []
        if isinstance(rag_result, dict) and "source_documents" in rag_result:
            for doc in rag_result["source_documents"]:
                if hasattr(doc, "page_content"):
                    retrieved_contexts.append(doc.page_content)
                elif isinstance(doc, dict) and "page_content" in doc:
                    retrieved_contexts.append(doc["page_content"])
                else:
                    retrieved_contexts.append(str(doc))
        
        print(f"✅ Sucesso para pergunta: {pergunta_id} ({len(retrieved_contexts)} contextos)")
        
    except TimeoutException as e:
        # Timeout específico - será retryado automaticamente pelo decorator
        raise TimeoutException(f"Timeout ao processar pergunta {pergunta_id}: {str(e)}")
    except Exception as e:
        # Outros erros - podem ser retryados se forem temporários
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['timeout', 'connection', 'network', 'temporary']):
            raise RetryableError(f"Erro retryável na pergunta {pergunta_id}: {str(e)}")
        else:
            # Erro não retryável
            raise Exception(f"Erro permanente na pergunta {pergunta_id}: {str(e)}")
    
    # Avaliar resposta
    expected_value = extract_monetary_value_fast(ground_truth_answer)
    generated_value = extract_monetary_value_fast(generated_answer)
    is_correct = False
    diff_valor = None
    
    if expected_value and generated_value:
        is_correct = (expected_value == generated_value)
        diff_valor = "exact" if is_correct else f"expected={expected_value}, generated={generated_value}"
    else:
        if ground_truth_answer and generated_answer and ground_truth_answer.strip().lower() == generated_answer.strip().lower():
            is_correct = True
            diff_valor = "exact text"
        else:
            diff_valor = "not exact"
    
    return pergunta_id, {
        "id_versao_pergunta": pergunta_id,
        "pergunta": question,
        "resposta_esperada": ground_truth_answer,
        "resposta_gerada": generated_answer.strip().replace('\n', ' '),
        "acerto": is_correct,
        "pdf": pdf,
        "extrato": extrato,
        "contextos_recuperados": retrieved_contexts,
        "diferenca_valor": diff_valor,
        "contextos_count": len(retrieved_contexts),
        "sistema": "final_processor_v3_no_signals",
        "timestamp": time.time()
    }


def process_batch_simple(qa_chain, batch_items, output_file):
    """
    Processa batch de forma simples com 1 worker - evita problemas de signals
    """
    print(f"📦 Processando batch de {len(batch_items)} itens (1 worker)")
    
    results = []
    stats = {"success": 0, "error": 0, "timeout": 0}
    
    # Usar apenas 1 worker para evitar problemas com signals
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submeter todas as tasks
        future_to_item = {
            executor.submit(evaluate_one_final, qa_chain, item): item 
            for item in batch_items
        }
        
        # Processar com timeout usando as_completed (mais simples)
        progress_bar = tqdm(total=len(batch_items), desc="Processando")
        
        try:
            # Timeout maior para o batch completo
            for future in cf.as_completed(future_to_item, timeout=len(batch_items) * PER_TASK_TIMEOUT):
                item = future_to_item[future]
                pergunta_id = item.get("id_versao_pergunta", "unknown")
                
                try:
                    pid, result = future.result(timeout=5)  # Deveria estar pronto
                    results.append(result)
                    stats["success"] += 1
                    progress_bar.set_postfix({"✅": stats["success"], "❌": stats["error"], "⏰": stats["timeout"]})
                    
                except Exception as e:
                    error_result = create_error_result(item, f"ERRO: {str(e)}")
                    results.append(error_result)
                    if "timeout" in str(e).lower():
                        stats["timeout"] += 1
                    else:
                        stats["error"] += 1
                
                progress_bar.update(1)
                
        except cf.TimeoutError:
            # Timeout do batch inteiro
            print(f"⏰ Timeout geral do batch após {len(batch_items) * PER_TASK_TIMEOUT}s")
            
            # Processar futures restantes como erro
            for future, item in future_to_item.items():
                if not future.done():
                    future.cancel()
                    error_result = create_error_result(item, "ERRO: Timeout geral do batch")
                    results.append(error_result)
                    stats["timeout"] += 1
                    progress_bar.update(1)
        
        progress_bar.close()
    
    # Escrever resultados no arquivo
    with open(output_file, 'a', encoding='utf-8') as jsonlfile:
        for result in results:
            jsonlfile.write(json.dumps(result, ensure_ascii=False) + '\n')
        jsonlfile.flush()
    
    print(f"📊 Batch concluído: {stats['success']} sucessos, {stats['error']} erros, {stats['timeout']} timeouts")
    return stats["success"], stats["error"], stats["timeout"]


def retry_empty_contexts(output_jsonl, qa_chain):
    """Retry casos com contextos vazios - função original melhorada"""
    if not os.path.exists(output_jsonl):
        print("❌ Arquivo de resultados não encontrado para retry")
        return
    
    with open(output_jsonl, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f if line.strip()]
    
    to_retry = [x for x in results if x.get('contextos_count', 0) == 0 and not x.get('acerto', False)]
    print(f"\n🔁 Retrying {len(to_retry)} cases with empty contextos...")
    
    if not to_retry:
        print("✅ Nenhum caso para retry")
        return
    
    updated = {x['id_versao_pergunta']: x for x in results}
    
    # Processar em batches pequenos para retry (sequencial para ser mais seguro)
    retry_batch_size = 5
    for i in range(0, len(to_retry), retry_batch_size):
        batch = to_retry[i:i + retry_batch_size]
        print(f"🔄 Retry batch {i//retry_batch_size + 1}/{(len(to_retry)-1)//retry_batch_size + 1}")
        
        for item in batch:
            try:
                # Processar um por vez no retry para máxima estabilidade
                pid, result = evaluate_one_final(qa_chain, {
                    "id_versao_pergunta": item["id_versao_pergunta"], 
                    "pergunta": item["pergunta"],
                    "resposta": item["resposta_esperada"],
                    "pdf": item.get("pdf", ""),
                    "extrato": item.get("extrato")
                })
                updated[pid] = result
                print(f"✅ Retry sucesso: {pid}")
            except Exception as e:
                print(f"❌ Retry falhou: {item.get('id_versao_pergunta')} - {str(e)}")
    
    # Reescrever arquivo com resultados atualizados
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for obj in updated.values():
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    
    print("✅ Retentativa dos contextos vazios concluída.")


def run_final_evaluation(input_file, output_file):
    """
    FUNÇÃO PRINCIPAL SEM SIGNALS - Corrige problemas de threading
    """
    print("🔧 VERSÃO SEM SIGNALS - CORRIGE PROBLEMA DE THREADING")
    print(f"🚀 Processando {input_file} → {output_file}")
    print(f"⚙️  Configurações: Workers={MAX_WORKERS}, Timeout={PER_TASK_TIMEOUT}s, Retries={MAX_ATTEMPTS}")
    print("📝 Usando threading timeout ao invés de signals")
    
    # Carregar dados
    evaluation_data = load_evaluation_data(input_file)
    if not evaluation_data:
        print(f"❌ Nenhum dado para processar.")
        return
    
    # Inicializar sistema RAG
    try:
        qa_chain = build_system()
    except Exception as e:
        print(f"❌ Erro ao inicializar sistema RAG: {str(e)}")
        return
    
    # Backup se arquivo existe
    if os.path.exists(output_file):
        import shutil
        backup_file = f"{output_file}.bak.{int(time.time())}"
        shutil.copy2(output_file, backup_file)
        print(f"💾 Backup criado: {backup_file}")
    
    # Loop principal de processamento
    processed_successfully, processed_failed, failed_retryable = load_existing_results_final(output_file)
    total_processed = len(processed_successfully) + len(processed_failed)
    run_count = 0
    
    while total_processed < TARGET_TOTAL_CASES:
        run_count += 1
        print(f"\n🔄 === RUN {run_count} === Processados: {total_processed}/{TARGET_TOTAL_CASES}")
        
        work_items, already_done = determine_work_final(
            evaluation_data, processed_successfully, processed_failed, failed_retryable
        )
        
        if not work_items:
            print("✅ Nenhum trabalho restante!")
            break
        
        # Processar próximo batch
        batch = work_items[:BATCH_SIZE]
        print(f"📦 Processando batch de {len(batch)} itens...")
        
        try:
            success, errors, timeouts = process_batch_simple(qa_chain, batch, output_file)
            
            # Recarregar resultados atualizados
            processed_successfully, processed_failed, failed_retryable = load_existing_results_final(output_file)
            total_processed = len(processed_successfully) + len(processed_failed)
            
            print(f"📈 Run {run_count} concluído:")
            print(f"   Total processados: {total_processed}")
            print(f"   Sucessos: {success}, Erros: {errors}, Timeouts: {timeouts}")
            
            if total_processed >= TARGET_TOTAL_CASES:
                print(f"🎊 Meta atingida! {total_processed} casos processados >= {TARGET_TOTAL_CASES}")
                break
            
            # Pausa entre runs para descanso do sistema
            if run_count % 3 == 0:
                print("⏳ Pausa de 30s para descanso do sistema...")
                time.sleep(30)
                
        except Exception as e:
            print(f"❌ Erro crítico no run {run_count}: {str(e)}")
            print("🔄 Continuando com próximo batch...")
            continue
    
    print(f"\n🏁 PROCESSAMENTO PRINCIPAL CONCLUÍDO")
    print(f"📊 Total processado: {total_processed} casos")
    
    # Retry casos com contextos vazios
    print(f"\n🔁 INICIANDO RETRY DE CONTEXTOS VAZIOS...")
    try:
        retry_empty_contexts(output_file, qa_chain)
    except Exception as e:
        print(f"❌ Erro no retry de contextos vazios: {str(e)}")
    
    print(f"\n✅ AVALIAÇÃO FINAL CONCLUÍDA!")
    print(f"📁 Resultados salvos em: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("❌ Uso incorreto!")
        print("✅ Uso correto: python final_eval_no_signals.py <input_base.jsonl> <output_results.jsonl>")
        print("\nExemplo:")
        print("python final_eval_no_signals.py base_b_extrato_final2.jsonl results.jsonl")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print("=" * 80)
    print("🚀 SCRIPT DE AVALIAÇÃO RAG - VERSÃO SEM SIGNALS")
    print("=" * 80)
    print(f"📁 Input:  {input_file}")
    print(f"📁 Output: {output_file}")
    print("🔧 Timeout via threading (não signals)")
    print("=" * 80)
    
    run_final_evaluation(input_file, output_file)