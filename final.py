import sys
import os
import json
import csv
import time
import gc
import torch
from itertools import product

# --- Verificação de FAISS-GPU ---
try:
    import faiss
    
    res = faiss.StandardGpuResources()
    print("✅ FAISS-GPU detectado e acessível! Os testes usarão a GPU.")
    # Opcional: Descomente para ver qual GPU está sendo usada
    # print(f"   -> Usando GPU ID: {res.getDevice()}")
except ImportError:
    print("⚠️ AVISO: A biblioteca 'faiss' não está instalada.")
    print("   -> Rode: pip install faiss-gpu")
    sys.exit() # Para o script se FAISS não estiver instalado
except AttributeError:
    print("❌ AVISO: FAISS-CPU detectado (ou erro na configuração da GPU).")
    print("   -> A indexação e busca serão MUITO lentas.")
    print("   -> Para usar a GPU, instale 'faiss-gpu':")
    print("      pip uninstall faiss-cpu")
    print("      pip install faiss-gpu")
    # Decida se quer parar o script ou continuar na CPU
    # sys.exit() # Descomente para parar se a GPU não for encontrada
except Exception as e:
    print(f"❌ ERRO Inesperado ao verificar FAISS-GPU: {e}")
    print("   -> Verifique sua instalação do CUDA e do FAISS.")
    # sys.exit() # Descomente para parar em caso de erro

# --- Configurações de Teste ---
# (Restante do seu código continua aqui)
# ...


# --- Configurações de Teste ---
#CHUNKS       = [400, 600, 800, 1000]
CHUNKS       = [600]
#VIZINHOS     = [0, 1]
VIZINHOS     = [1]
#K_VALUES     = [1, 3, 5, 7]
K_VALUES     = [5]
MODELS       = [{'name':'llama3','url':'http://localhost:11434','model':'llama3.1:8b-32k'}]
#MODELS       = [{'name': 'llama4', 'url': 'http://164.41.75.221:11434', 'model': 'llama4:latest'}]
#BASES        = [
 #   {'name':'a','file':'base_a_objeto.jsonl'},
  #  {'name':'b','file':'base_b_extrato_final.jsonl'}]
BASES        = [
    {'name':'b','file':'dataset_RAG_BaseB_UNIAO_FINAL.jsonl'}
]

RESULTS_DIR  = "dataset_RAG_VALIDADO_UNIAOCORRETO_jsonl"
CSV_FILE     = "resultados_finais_FINALE.csv"
VECTOR_STORE_DIR = "data/vector_store"
PDF_DIRECTORY    = "data/pdfs/contratos_validado"

def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def reload_modules():
    for m in ['src.llm_interface','src.vector_store_manager','src.rag_chain_builder','final_eval']:
        if m in sys.modules: del sys.modules[m]

def run_test(chunk, viz, k, model, base_info):
    clear_gpu()
    
    faiss_index_name = f"faiss_index_chunk{chunk}_base_validado"
    faiss_index_path = os.path.join(VECTOR_STORE_DIR, faiss_index_name)

    reload_modules()
    from src.vector_store_manager import get_vector_store
    from final_eval import run_final_evaluation

    out = f"{chunk}_{viz}_{k}_{base_info['name']}_{model['name']}.jsonl"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outfile = os.path.join(RESULTS_DIR, out)

    print(f"🚀 Testando: {out}")
    vector_store = get_vector_store(
        faiss_index_path=faiss_index_path,
        pdf_directory=PDF_DIRECTORY,
        chunk_size=chunk,
        chunk_overlap=0,
        force_recreate=False
    )
    
    run_final_evaluation(
        input_file=base_info['file'],
        output_file=outfile,
        vector_store=vector_store,
        chunk_size=chunk,
        k=k,
        vizinhos=viz,
        pdf_directory=PDF_DIRECTORY
    )
    clear_gpu()

    acertos = 0
    total   = 0
    if os.path.exists(outfile):
        with open(outfile, 'r', encoding='utf-8') as f:
            for line in f:
                j = json.loads(line)
                total += 1
                if j.get('acerto', False):
                    acertos += 1
    else:
        print(f"❌ Arquivo de resultado não encontrado: {outfile}")
        return

    erros    = total - acertos
    acuracia = (acertos/total*100) if total else 0.0

    write_header = not os.path.exists(CSV_FILE)
    with open(CSV_FILE,'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['arquivo_jsonl','chunk','vizinho','k','modelo','acertos','erros','acuracia'])
        w.writerow([out.replace('.jsonl',''), chunk, viz, k, model['name'], acertos, erros, f"{acuracia:.2f}%"])

def main():
    for chunk, viz, k, model, base_info in product(CHUNKS, VIZINHOS, K_VALUES, MODELS, BASES):
        fname = f"{chunk}_{viz}_{k}_{base_info['name']}_{model['name']}.jsonl"
        if os.path.exists(os.path.join(RESULTS_DIR, fname)):
            continue
        run_test(chunk, viz, k, model, base_info)

if __name__=="__main__":
    main()
