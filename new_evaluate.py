# new_evaluate_fixed.py
# Script de avaliação usando componentes CORRIGIDOS (PARALELO + RETRY + DEDUP) - versão corrigida

import sys
import os
import json
import re
import time
import math
import threading
import concurrent.futures as cf
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# USAR componentes CORRIGIDOS
from src.llm_interface import get_llm
from src.vector_store_manager import get_vector_store
from src.rag_chain_builder import build_rag_chain  # usa sua versão híbrida
from src.config import RETRIEVER_SEARCH_K

# --- Configurações da Avaliação ---
DATASET_FILE_PATH = "./benchmark_final_valor.jsonl"
RESULTS_JSONL_PATH = "./evaluation_results_hibrida_800_3_no_neigh.jsonl"
NUM_QUESTIONS_TO_TEST = 1000  # Para teste inicial

# --- Configurações de Concorrência/Retry ---
MAX_WORKERS = min(16, (os.cpu_count() or 4) * 2)   # limitar agressividade
PER_TASK_TIMEOUT = 60.0 * 5                         # segundos para considerar "demorado"
MAX_ATTEMPTS = 3                                    # reenvios máximos por row
BACKOFF_BASE = 1.8                                  # backoff exponencial simples
CHECK_INTERVAL = 1.0                                # varredura de timeouts
# Tempo máximo total antes de desistir de um row (ex.: 3 tentativas × 5 min = 15 min)
GIVE_UP_AFTER = PER_TASK_TIMEOUT * MAX_ATTEMPTS

def extract_monetary_value(text: str) -> str or None:
    """Extrai e normaliza um valor monetário de uma string."""
    if not isinstance(text, str):
        return None
    match = re.search(r'R\$\s*([\d\.,]+)', text)
    if match:
        value_str = match.group(1).strip()
        normalized_value = value_str.replace('.', '').replace(',', '.')
        try:
            return str(float(normalized_value))
        except ValueError:
            return None
    return None

def load_evaluation_data(file_path: str):
    data = []
    if not os.path.exists(file_path):
        print(f"ERRO: Arquivo de dataset não encontrado em '{file_path}'")
        return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"AVISO: Pulando linha mal formatada: {line.strip()}")
    return data

def build_system():
    """Inicializa vector store, LLM e chain RAG."""
    print("\n🔧 Inicializando sistema RAG CORRIGIDO...")
    # 1) Vector store
    print("1️⃣ Criando/carregando vector store CORRIGIDO...")
    vector_store = get_vector_store(force_recreate=False)
    # 2) LLM
    print("2️⃣ Inicializando LLM...")
    llm = get_llm()
    # 3) RAG (híbrido)
    print("3️⃣ Construindo cadeia RAG CORRIGIDA (híbrida)...")
    qa_chain = build_rag_chain(llm, vector_store)  # mantém sua versão híbrida
    if not all([vector_store, llm, qa_chain]):
        raise RuntimeError("Falha ao inicializar componentes do RAG")
    print("✅ Sistema RAG CORRIGIDO inicializado com sucesso!")
    return qa_chain

def evaluate_one(qa_chain, item):
    """
    Executa uma pergunta, constrói o log_row e retorna (row_id, log_row).
    row_id é único por linha do dataset e dirige dedupe/progresso.
    """
    row_id = item.get("_row_id")  # sempre definido
    id_pergunta = item.get("id_versao_pergunta")
    question = item.get("pergunta")
    ground_truth_answer = item.get("resposta") or item.get("answer")
    pdf = item.get("pdf", "")
    extrato = item.get("extrato")

    if not question or not ground_truth_answer:
        log_row = {
            "id_versao_pergunta": id_pergunta,
            "pergunta": question,
            "resposta_esperada": ground_truth_answer,
            "resposta_gerada": "ERRO: Pergunta/Resposta inválida no dataset",
            "acerto": False,
            "pdf": pdf,
            "extrato": extrato,
            "contextos_recuperados": [],
            "diferenca_valor": None,
            "contextos_count": 0,
            "sistema": "corrigido_parallel"
        }
        return row_id, log_row

    # Chamada ao LLM/RAG
    try:
        result = qa_chain.invoke({"query": question})
        # Compatibilidade: algumas chains retornam dict; outras, string
        if isinstance(result, dict):
            generated_answer = result.get("result", "") or result.get("answer", "")
        else:
            generated_answer = str(result or "")
        if not generated_answer:
            generated_answer = "ERRO: Sem resultado"
    except Exception as e:
        generated_answer = f"ERRO NA EXECUÇÃO: {e}"
        result = None

    # Contextos
    retrieved_contexts = []
    if isinstance(result, dict) and "source_documents" in result:
        try:
            retrieved_contexts = [doc.page_content for doc in result["source_documents"]]
        except Exception:
            retrieved_contexts = []

    # Avaliação por valor monetário
    expected_value = extract_monetary_value(ground_truth_answer)
    generated_value = extract_monetary_value(generated_answer)
    is_correct = False
    diff_valor = None
    if expected_value is not None and generated_value is not None:
        try:
            diff_valor = abs(float(expected_value) - float(generated_value))
            is_correct = diff_valor < 0.1
        except Exception:
            pass

    log_row = {
        "id_versao_pergunta": id_pergunta,
        "pergunta": question,
        "resposta_esperada": ground_truth_answer,
        "resposta_gerada": generated_answer.strip().replace('\n', ' '),
        "acerto": is_correct,
        "pdf": pdf,
        "extrato": extrato,
        "contextos_recuperados": retrieved_contexts,
        "diferenca_valor": diff_valor,
        "contextos_count": len(retrieved_contexts),
        "sistema": "corrigido_parallel"
    }
    return row_id, log_row

def make_timeout_log_row(item, msg):
    """Cria um log_row de timeout para ‘desistir’ de uma linha após tentativas esgotadas."""
    id_pergunta = item.get("id_versao_pergunta")
    question = item.get("pergunta")
    ground_truth_answer = item.get("resposta") or item.get("answer")
    pdf = item.get("pdf", "")
    extrato = item.get("extrato")
    return {
        "id_versao_pergunta": id_pergunta,
        "pergunta": question,
        "resposta_esperada": ground_truth_answer,
        "resposta_gerada": f"ERRO: {msg}",
        "acerto": False,
        "pdf": pdf,
        "extrato": extrato,
        "contextos_recuperados": [],
        "diferenca_valor": None,
        "contextos_count": 0,
        "sistema": "corrigido_parallel"
    }

def run_fixed_evaluation():
    print("🛠️ === AVALIAÇÃO RAG CORRIGIDA (PARALELA) ===")
    evaluation_data = load_evaluation_data(DATASET_FILE_PATH)
    if not evaluation_data:
        return

    if NUM_QUESTIONS_TO_TEST is not None:
        print(f"⚠️ ATENÇÃO: Testando apenas as primeiras {NUM_QUESTIONS_TO_TEST} perguntas")
        evaluation_data = evaluation_data[:NUM_QUESTIONS_TO_TEST]

    # Normalize: crie um row_id único por linha para dedupe/progresso
    normalized = []
    for idx, item in enumerate(evaluation_data):
        # Preserve o id original apenas para log; não use para dedupe/progresso
        item = dict(item)  # evitar mutar referência externa
        item["_row_id"] = f"row-{idx}"
        normalized.append(item)
    evaluation_data = normalized

    total = len(evaluation_data)
    print(f"📊 Dataset carregado: {total} linhas (1:1 com progresso)")

    # Inicializar sistema
    qa_chain = build_system()

    # Estado de execução
    completed_ids = set()         # row_ids concluídos (inclui desistências)
    attempts_by_id = {}           # row_id -> tentativas realizadas
    first_submit_time = {}        # row_id -> tempo do primeiro submit (para 'give up')
    lock = threading.Lock()       # proteger métricas
    correct_answers = 0
    results_data = []

    print(f"\n🔍 Iniciando avaliação CORRIGIDA (paralela) de {total} linhas...")

    with open(RESULTS_JSONL_PATH, 'w', encoding='utf-8') as jsonlfile:
        with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = set()
            # meta: Future -> {"start": t, "rid": row_id, "attempt": n, "item": item}
            meta = {}

            def submit_item(item, attempt):
                rid = item["_row_id"]
                future = executor.submit(evaluate_one, qa_chain, item)
                meta[future] = {"start": time.time(), "rid": rid, "attempt": attempt, "item": item}
                futures.add(future)
                if attempt == 1 and rid not in first_submit_time:
                    first_submit_time[rid] = meta[future]["start"]

            # Submeter tudo uma vez
            for item in evaluation_data:
                rid = item["_row_id"]
                attempts_by_id[rid] = 1
                submit_item(item, 1)

            pbar = tqdm(total=total, desc="Avaliando (paralelo)")
            try:
                while len(completed_ids) < total:
                    done, not_done = cf.wait(futures, timeout=CHECK_INTERVAL, return_when=cf.FIRST_COMPLETED)

                    # Processar concluídos
                    for future in list(done):
                        info = meta.pop(future, None)
                        futures.discard(future)
                        try:
                            rid, log_row = future.result()  # se deu erro, levanta aqui
                        except Exception as e:
                            # Tratar como falha transitória: reenvio se cabível
                            if info:
                                rid = info["rid"]
                                item = info["item"]
                                attempt = attempts_by_id.get(rid, 1)
                                if rid not in completed_ids and attempt < MAX_ATTEMPTS:
                                    attempts_by_id[rid] = attempt + 1
                                    backoff = BACKOFF_BASE ** attempt
                                    time.sleep(min(2.0, backoff))  # pequeno backoff
                                    submit_item(item, attempt + 1)
                            continue

                        # Deduplicar por row_id
                        with lock:
                            if rid in completed_ids:
                                continue
                            completed_ids.add(rid)

                            if log_row.get("acerto"):
                                correct_answers += 1
                            results_data.append(log_row)
                            jsonlfile.write(json.dumps(log_row, ensure_ascii=False) + "\n")
                            jsonlfile.flush()
                            pbar.update(1)

                    # Reenviar timeouts (ainda não concluídos) ou desistir após GIVE_UP_AFTER
                    now = time.time()
                    for future in list(not_done):
                        info = meta.get(future)
                        if not info:
                            continue
                        rid = info["rid"]
                        start = info["start"]
                        elapsed = now - start
                        total_elapsed = now - first_submit_time.get(rid, start)
                        if rid in completed_ids:
                            # Se já completou (por outro future) pare de rastrear este
                            meta.pop(future, None)
                            futures.discard(future)
                            try:
                                future.cancel()
                            except Exception:
                                pass
                            continue

                        attempt = attempts_by_id.get(rid, 1)

                        # Caso 1: timeout de tentativa atual -> reenvio se ainda houver tentativas
                        if elapsed > PER_TASK_TIMEOUT and attempt < MAX_ATTEMPTS:
                            item = info["item"]
                            attempts_by_id[rid] = attempt + 1
                            submit_item(item, attempt + 1)
                            # Não removemos o antigo future; resultado tardio será ignorado pelo dedupe

                        # Caso 2: tentativas esgotadas e tempo total excedido -> desistir e registrar erro
                        elif attempt >= MAX_ATTEMPTS and total_elapsed > GIVE_UP_AFTER:
                            with lock:
                                if rid not in completed_ids:
                                    completed_ids.add(rid)
                                    timeout_log = make_timeout_log_row(
                                        info["item"],
                                        f"Timeout excedido após {attempt} tentativas e ~{int(total_elapsed)}s"
                                    )
                                    results_data.append(timeout_log)
                                    jsonlfile.write(json.dumps(timeout_log, ensure_ascii=False) + "\n")
                                    jsonlfile.flush()
                                    pbar.update(1)
                            # Deixar de rastrear este future
                            meta.pop(future, None)
                            futures.discard(future)
                            try:
                                future.cancel()
                            except Exception:
                                pass

                pbar.close()

            except KeyboardInterrupt:
                print("\n🛑 Interrompido por teclado; aguardando tarefas em curso finalizarem com timeout curto...")
                done, _ = cf.wait(futures, timeout=5, return_when=cf.FIRST_COMPLETED)
                for future in done:
                    info = meta.get(future)
                    try:
                        rid, log_row = future.result()
                        with lock:
                            if rid not in completed_ids:
                                completed_ids.add(rid)
                                if log_row.get("acerto"):
                                    correct_answers += 1
                                results_data.append(log_row)
                                jsonlfile.write(json.dumps(log_row, ensure_ascii=False) + "\n")
                                jsonlfile.flush()
                    except Exception:
                        pass

    # Resultados finais
    total_questions = total
    accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0.0
    print(f"\n🎉 === RESULTADOS FINAIS CORRIGIDOS ===")
    print(f"📊 Total de linhas: {total_questions}")
    print(f"✅ Respostas corretas: {correct_answers}")
    print(f"📈 Acurácia: {accuracy:.2f}%")
    print(f"💾 Resultados salvos em: {RESULTS_JSONL_PATH}")

    # Estatísticas adicionais
    contexts_stats = [r.get("contextos_count", 0) for r in results_data]
    if contexts_stats:
        avg_contexts = sum(contexts_stats) / len(contexts_stats)
        print(f"📋 Contextos médios por pergunta: {avg_contexts:.1f}")

    # Análise de erros
    errors = [r for r in results_data if not r.get("acerto")]
    if errors:
        print(f"❌ {len(errors)} erros encontrados")
        print("Primeiros 3 erros:")
        for i, error in enumerate(errors[:3]):
            print(f"  {i+1}. {str(error.get('pergunta', ''))[:50]}...")
            print(f"     Esperado: {error.get('resposta_esperada')}")
            print(f"     Gerado: {str(error.get('resposta_gerada', ''))[:60]}...")

if __name__ == "__main__":
    run_fixed_evaluation()
