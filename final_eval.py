import sys
import os
import json
import re
import time
import concurrent.futures as cf
from tqdm import tqdm
from collections import defaultdict

# --- CONFIGURAÇÕES ---
MAX_WORKERS = 4
PER_TASK_TIMEOUT = 180
MAX_ATTEMPTS = 2
TARGET_TOTAL_CASES = 294
BATCH_SIZE = 50

def extract_monetary_value_fast(text: str) -> str or None:
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
                    "Exception", "TimeoutError"
                ]):
                    failed_retryable[pergunta_id] = result
                else:
                    processed_failed[pergunta_id] = result
            except json.JSONDecodeError:
                continue
    return processed_successfully, processed_failed, failed_retryable

def determine_work_final(evaluation_data, processed_successfully, processed_failed, failed_retryable):
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

def evaluate_one_final(qa_chain, item):
    pergunta_id = item.get("id_versao_pergunta")
    question = item.get("pergunta")
    ground_truth_answer = item.get("resposta") or item.get("answer")
    pdf = item.get("pdf", "")
    extrato = item.get("extrato")
    if not question or not ground_truth_answer:
        return pergunta_id, {
            "id_versao_pergunta": pergunta_id,
            "pergunta": question or "",
            "resposta_esperada": ground_truth_answer or "",
            "resposta_gerada": "ERRO: Dados inválidos",
            "acerto": False,
            "pdf": pdf,
            "extrato": extrato,
            "contextos_recuperados": [],
            "diferenca_valor": "Dados inválidos",
            "contextos_count": 0,
            "sistema": "final_processor",
            "timestamp": time.time()
        }
    try:
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
            generated_answer = "ERRO: LLM retornou vazio"
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
    except Exception as e:
        return pergunta_id, {
            "id_versao_pergunta": pergunta_id,
            "pergunta": question,
            "resposta_esperada": ground_truth_answer,
            "resposta_gerada": f"ERRO CRÍTICO: {str(e)}",
            "acerto": False,
            "pdf": pdf,
            "extrato": extrato,
            "contextos_recuperados": [],
            "diferenca_valor": None,
            "contextos_count": 0,
            "sistema": "final_processor",
            "timestamp": time.time()
        }
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
        "sistema": "final_processor",
        "timestamp": time.time()
    }

def retry_empty_contexts(output_jsonl, qa_chain):
    with open(output_jsonl, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f]
    to_retry = [x for x in results if x.get('contextos_count', 0) == 0]
    print(f"\n🔁 Retrying {len(to_retry)} cases with empty contextos...")
    updated = {x['id_versao_pergunta']: x for x in results}
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(evaluate_one_final, qa_chain, item): item
            for item in to_retry
        }
        for future in tqdm(cf.as_completed(futures), total=len(to_retry), desc="Retentando contextos vazios"):
            pid, result = future.result()
            updated[pid] = result
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for obj in updated.values():
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    print("✅ Retentativa dos contextos vazios concluída.")

def run_final_evaluation(input_file, output_file):
    print(f"🚀 Processando {input_file} → {output_file}")
    evaluation_data = load_evaluation_data(input_file)
    if not evaluation_data:
        print(f"❌ Nenhum dado para processar.")
        return
    qa_chain = build_system()
    if os.path.exists(output_file):
        import shutil
        shutil.copy2(output_file, output_file + ".bak")
        print(f"💾 Backup criado: {output_file}.bak")
    processed_successfully, processed_failed, failed_retryable = load_existing_results_final(output_file)
    total_processed = len(processed_successfully) + len(processed_failed)
    run_count = 0

    while total_processed < TARGET_TOTAL_CASES:
        run_count += 1
        print(f"\n🔄 === RUN {run_count} === Já processados: {total_processed}")
        work_items, already_done = determine_work_final(evaluation_data, processed_successfully, processed_failed, failed_retryable)
        if not work_items:
            print("✅ Nenhum trabalho restante!")
            break
        batch = work_items[:BATCH_SIZE]
        with open(output_file, 'w', encoding='utf-8') as jsonlfile:
            for result in list(processed_successfully.values()) + list(processed_failed.values()):
                jsonlfile.write(json.dumps(result, ensure_ascii=False) + '\n')
            with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(evaluate_one_final, qa_chain, item): item
                    for item in batch
                }
                pbar = tqdm(total=len(batch), desc=f"Processando batch")
                for future in cf.as_completed(futures):
                    pid, result = future.result()
                    jsonlfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                    jsonlfile.flush()
                    pbar.update(1)
                pbar.close()
        processed_successfully, processed_failed, failed_retryable = load_existing_results_final(output_file)
        total_processed = len(processed_successfully) + len(processed_failed)
        print(f"📈 Run {run_count} done: Total {total_processed}")
        if total_processed >= TARGET_TOTAL_CASES:
            print(f"🎊 Meta atingida! {total_processed} casos processados >= {TARGET_TOTAL_CASES}")
            break

    # RETRY CASES WITH EMPTY CONTEXT AT END
    retry_empty_contexts(output_file, qa_chain)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python final_eval.py <input_base.jsonl> <output_results.jsonl>")
        sys.exit(1)
    run_final_evaluation(sys.argv[1], sys.argv[2])
