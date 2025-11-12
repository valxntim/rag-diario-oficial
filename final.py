"""
🔬 TESTE COM GRID SEARCH - MÚLTIPLAS CONFIGURAÇÕES
===================================================
Script que executa avaliações com DIFERENTES combinações de:
- Tamanho de chunks (400, 600, 800, 1000)
- Com/sem neighbor retriever (vizinhos 0 ou 1)
- Top K a recuperar (1, 3, 5, 7)
- Datasets diferentes (base_a, base_b)

Cada combinação é testada INDEPENDENTEMENTE, gerando resultados em CSV.
Usa GPU quando possível (FAISS-GPU detecta automaticamente).
"""

import sys
import os
import json
import csv
import time
import gc
import torch
from itertools import product  # Para gerar todas as combinações


# ========== VERIFICAÇÃO DE FAISS-GPU ==========
# Detecta se FAISS com GPU está disponível
try:
    import faiss
    
    # Tenta acessar recursos GPU do FAISS
    res = faiss.StandardGpuResources()
    print("✅ FAISS-GPU detectado e acessível! Os testes usarão a GPU.")
    # Opcional: Descomente para ver qual GPU está sendo usada
    # print(f"   -> Usando GPU ID: {res.getDevice()}")
except ImportError:
    # FAISS não está instalado
    print("⚠️ AVISO: A biblioteca 'faiss' não está instalada.")
    print("   -> Rode: pip install faiss-gpu")
    sys.exit()  # Para o script (FAISS é essencial)
except AttributeError:
    # FAISS está instalado, mas é versão CPU
    print("❌ AVISO: FAISS-CPU detectado (ou erro na configuração da GPU).")
    print("   -> A indexação e busca serão MUITO lentas.")
    print("   -> Para usar a GPU, instale 'faiss-gpu':")
    print("      pip uninstall faiss-cpu")
    print("      pip install faiss-gpu")
    # Decida se quer parar o script ou continuar na CPU
    # sys.exit()  # Descomente para parar se a GPU não for encontrada
except Exception as e:
    # Erro inesperado ao checkar FAISS
    print(f"❌ ERRO Inesperado ao verificar FAISS-GPU: {e}")
    print("   -> Verifique sua instalação do CUDA e do FAISS.")
    # sys.exit()  # Descomente para parar em caso de erro


# ========== CONFIGURAÇÕES DE TESTE ==========
# Define os parâmetros a testar (grid search)

# Tamanhos de chunks a testar (quanto maior, mais contexto por chunk)
CHUNKS       = [400, 600, 800, 1000]

# Com/sem neighbor retriever
# 0 = sem (mais rápido, menos contexto)
# 1 = com (mais lento, melhor contexto)
VIZINHOS     = [0, 1]

# Número de chunks a recuperar (top-K)
K_VALUES     = [1, 3, 5, 7]

# Modelos LLM a testar (pode ter múltiplos se quiser comparar)
MODELS       = [{'name':'llama3','url':'http://localhost:11434','model':'llama3.1:8b-32k'}]

# Datasets de benchmark a testar
BASES        = [
    {'name':'a','file':'base_a_objeto.jsonl'},
    {'name':'b','file':'base_b_extrato_final.jsonl'}
]

# Diretório onde salvar resultados de CADA teste
RESULTS_DIR  = "resultados_valido_base_menor"

# Arquivo CSV consolidado com resumo de TODOS os testes
CSV_FILE     = "resultados_finais_base_pequena.csv"

# Diretórios padrão (usar paths de config.py em produção)
VECTOR_STORE_DIR = "data/vector_store"
PDF_DIRECTORY    = "data/pdfs/contratos_validado"


# ========== UTILITÁRIOS DE LIMPEZA ==========

def clear_gpu():
    """
    Limpa cache e memória da GPU entre testes.
    Evita Out-of-Memory quando rodando múltiplos testes sequencialmente.
    """
    # Se CUDA está disponível, limpa cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Força coleta de lixo (garbage collection)
    gc.collect()


def reload_modules():
    """
    Remove módulos do cache do Python para forçar reimportação.
    Necessário para testar diferentes configurações sequencialmente.
    
    Sem isso, Python reutilizaria o módulo anterior com suas configurações.
    """
    for m in ['src.llm_interface','src.vector_store_manager','src.rag_chain_builder','final_eval']:
        if m in sys.modules: 
            del sys.modules[m]


# ========== FUNÇÃO DE TESTE INDIVIDUAL ==========

def run_test(chunk, viz, k, model, base_info):
    """
    Executa UM teste com UMA combinação de parâmetros.
    
    Processo:
    1. Limpa GPU
    2. Determina nome do índice FAISS para este chunk_size
    3. Carrega/cria vector store
    4. Executa avaliação contra o dataset
    5. Coleta resultados e escreve em CSV
    
    Argumentos:
        chunk: Tamanho de chunks (ex: 600)
        viz: Com/sem vizinhos (0 ou 1)
        k: Top K a recuperar (ex: 5)
        model: Dict com informações do modelo
        base_info: Dict com informações do dataset
    """
    # Limpa GPU para evitar memory leak
    clear_gpu()
    
    # ========== SETUP DO TESTE ==========
    # Determina o nome do índice FAISS baseado no chunk_size
    faiss_index_name = f"faiss_index_chunk{chunk}_base_validado"
    faiss_index_path = os.path.join(VECTOR_STORE_DIR, faiss_index_name)

    # Força reimportação dos módulos src/ para usar nova config
    reload_modules()
    
    # Importa funções necessárias APÓS reload
    from src.vector_store_manager import get_vector_store
    from final_eval import run_final_evaluation

    # ========== NOME DO ARQUIVO DE SAÍDA ==========
    # Cada teste gera seu próprio arquivo JSONL com resultados
    out = f"{chunk}_{viz}_{k}_{base_info['name']}_{model['name']}.jsonl"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outfile = os.path.join(RESULTS_DIR, out)

    print(f"🚀 Testando: {out}")
    
    # ========== CARREGA OU CRIA VECTOR STORE ==========
    # Se índice não existe, cria novo (força_recreate = True)
    # Se existe, apenas carrega (força_recreate = False)
    vector_store = get_vector_store(
        faiss_index_path=faiss_index_path,
        pdf_directory=PDF_DIRECTORY,
        chunk_size=chunk,
        chunk_overlap=0,
        force_recreate=not os.path.exists(faiss_index_path)
        # force_recreate=False  # Descomente para sempre reusar índice existente
    )
    
    # ========== EXECUTA AVALIAÇÃO ==========
    # Processa o dataset completo contra este modelo/configuração
    run_final_evaluation(
        input_file=base_info['file'],           # Dataset JSONL
        output_file=outfile,                    # Onde salvar resultados
        vector_store=vector_store,              # Índice FAISS
        chunk_size=chunk,                       # Config de chunks
        k=k,                                    # Top K
        vizinhos=viz,                           # Com/sem neighbors
        pdf_directory=PDF_DIRECTORY
    )
    
    # Limpa GPU novamente após teste
    clear_gpu()

    # ========== COLETA RESULTADOS ==========
    # Lê arquivo JSONL de resultados e conta acertos/erros
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
        return  # Sai da função se não achou resultados

    # ========== CALCULA MÉTRICAS ==========
    erros    = total - acertos
    acuracia = (acertos/total*100) if total else 0.0

    # ========== SALVA RESULTADOS EM CSV ==========
    # Escreve uma linha no CSV consolidado
    write_header = not os.path.exists(CSV_FILE)  # Escreve header na 1ª linha
    with open(CSV_FILE,'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if write_header:
            # Header com nomes das colunas
            w.writerow(['arquivo_jsonl','chunk','vizinho','k','modelo','acertos','erros','acuracia'])
        # Dados do teste
        w.writerow([
            out.replace('.jsonl',''),  # Nome do arquivo (sem extensão)
            chunk,                     # Tamanho de chunks
            viz,                       # Com/sem vizinhos
            k,                         # Top K
            model['name'],             # Nome do modelo
            acertos,                   # Número de acertos
            erros,                     # Número de erros
            f"{acuracia:.2f}%"         # Acurácia formatada
        ])


# ========== FUNÇÃO PRINCIPAL ==========

def main():
    """
    Executa TODOS os testes combinando todos os parâmetros.
    
    Gera todas as combinações usando itertools.product:
    Ex: Se CHUNKS=[400,600], VIZINHOS=[0,1], K=[1,5]
    → 2 × 2 × 2 = 8 combinações
    
    Cada combinação é um teste independente.
    """
    # Gera TODAS as combinações dos parâmetros
    for chunk, viz, k, model, base_info in product(CHUNKS, VIZINHOS, K_VALUES, MODELS, BASES):
        # Monta nome do arquivo de resultado esperado
        fname = f"{chunk}_{viz}_{k}_{base_info['name']}_{model['name']}.jsonl"
        
        # Se resultado já existe, PULA este teste (evita reprocessar)
        if os.path.exists(os.path.join(RESULTS_DIR, fname)):
            print(f"⏭️ Pulando {fname} (já existe)")
            continue
        
        # Executa o teste
        run_test(chunk, viz, k, model, base_info)


# ========== BLOCO PRINCIPAL ==========
if __name__=="__main__":
    main()