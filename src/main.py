# Arquivo: src/main.py
# Versão Final Simplificada - Sem wrappers

# Usamos imports relativos porque estamos dentro do pacote 'src'
from .llm_interface import get_ollama_llm, get_specialized_embeddings
from .vector_store_manager import get_vector_store
from .rag_chain_builder import build_rag_chain

# Imports de bibliotecas externas
import json
import pandas as pd
from tqdm import tqdm
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset
import os

# --- Configurações da Avaliação (sem alterações) ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATASET_FILE_PATH = os.path.join(PROJECT_ROOT, "dataset_enriquecido.jsonl")
RESULTS_CSV_PATH = os.path.join(PROJECT_ROOT, "evaluation_results_final.csv")
NUM_QUESTIONS_TO_TEST = 100

def load_evaluation_data(file_path: str) -> list[dict]:
    # ... (sem alterações aqui) ...
    data = []
    if not os.path.exists(file_path):
        print(f"ERRO: Arquivo de dataset não encontrado em '{file_path}'")
        return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"AVISO: Pulando linha mal formatada no dataset: {line.strip()}")
    return data

def run_evaluation_task():
    """Esta função contém a lógica de avaliação, passando os modelos diretamente para o RAGAs."""
    print("--- Iniciando Avaliação Avançada do Sistema RAG com RAGAs ---")
    
    evaluation_data = load_evaluation_data(DATASET_FILE_PATH)
    if not evaluation_data:
        return
    if NUM_QUESTIONS_TO_TEST is not None:
        print(f"ATENÇÃO: Testando as primeiras {NUM_QUESTIONS_TO_TEST} perguntas do dataset.")
        evaluation_data = evaluation_data[:NUM_QUESTIONS_TO_TEST]

    # Carrega os modelos uma vez
    print("\nInicializando o sistema RAG...")
    vector_store = get_vector_store()
    llm_for_rag = get_ollama_llm()
    qa_chain = build_rag_chain(llm_for_rag, vector_store)
    print("Sistema RAG inicializado com sucesso.")

    # Coleta de respostas
    print(f"\nColetando respostas para {len(evaluation_data)} perguntas...")
    results_for_ragas = []
    for item in tqdm(evaluation_data, desc="Processando Perguntas"):
        question = item.get("question")
        ground_truth = item.get("answer")
        if not question or not ground_truth:
            continue
        try:
            result = qa_chain.invoke({"query": question})
            answer = result.get("result", "")
            contexts = [doc.page_content for doc in result.get("source_documents", [])]
        except Exception as e:
            answer = f"ERRO NA EXECUÇÃO: {e}"
            contexts = []
        results_for_ragas.append({ "question": question, "answer": answer, "contexts": contexts, "ground_truth": ground_truth })

    # Executa a avaliação RAGAs
    print("\nPreparando dados e executando a avaliação com RAGAs...")
    dataset = Dataset.from_list(results_for_ragas)
    
    # --- CHAMADA SIMPLIFICADA E CORRIGIDA ---
    ragas_result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm_for_rag, # Passamos o objeto LangChain diretamente
        embeddings=get_specialized_embeddings(), # Passamos o objeto LangChain diretamente
        raise_exceptions=False
    )
    # ------------------------------------
    
    print("Avaliação RAGAs concluída.")
    ragas_df = ragas_result.to_pandas()
    print("\n--- Resultados Médios da Avaliação RAGAs ---")
    print(ragas_df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean().round(3))
    print(f"\nSalvando resultados detalhados em '{RESULTS_CSV_PATH}'...")
    ragas_df.to_csv(RESULTS_CSV_PATH, index=False, encoding='utf-8-sig')
    print("Resultados salvos com sucesso.")