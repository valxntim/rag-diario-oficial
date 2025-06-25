# src/main.py
# Versão Final, Otimizada e Corrigida

# Imports de bibliotecas externas e locais
import os
import json
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# Imports dos seus módulos, agora com a nova função para o RAGAs
from .llm_interface import get_ollama_llm, get_ragas_llm
from .vector_store_manager import get_vector_store, get_embedding_model_for_processing
from .rag_chain_builder import build_rag_chain


# --- Configurações da Avaliação ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATASET_FILE_PATH = os.path.join(PROJECT_ROOT, "dataset_enriquecido.jsonl")
RESULTS_CSV_PATH = os.path.join(PROJECT_ROOT, "evaluation_results_final.csv")
NUM_QUESTIONS_TO_TEST = 1000

def load_evaluation_data(file_path: str) -> list[dict]:
    """Carrega os dados de avaliação de um arquivo .jsonl."""
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
    """Função principal que orquestra a inicialização do RAG e a avaliação com RAGAs."""
    print("--- Iniciando Avaliação Avançada do Sistema RAG com RAGAs ---")
    
    evaluation_data = load_evaluation_data(DATASET_FILE_PATH)
    if not evaluation_data:
        return
    if NUM_QUESTIONS_TO_TEST is not None:
        print(f"ATENÇÃO: Testando as primeiras {NUM_QUESTIONS_TO_TEST} perguntas do dataset.")
        evaluation_data = evaluation_data[:NUM_QUESTIONS_TO_TEST]

    # 1. Inicializa o sistema RAG completo com o LLM potente
    print("\nInicializando o sistema RAG...")
    vector_store = get_vector_store()
    llm_for_rag = get_ollama_llm() # Carrega o 'llama4:latest' para as respostas
    qa_chain = build_rag_chain(llm_for_rag, vector_store)
    
    if not qa_chain:
        print("ERRO FATAL: A cadeia RAG não pôde ser construída. Abortando a avaliação.")
        return
        
    print("Sistema RAG inicializado com sucesso.")

    # 2. "Aquece" a cadeia RAG para evitar erro na primeira chamada da GPU
    print("\nAquecendo o modelo da cadeia RAG para evitar erro na primeira pergunta...")
    try:
        _ = qa_chain.invoke("Pergunta de aquecimento para a GPU.")
        print("Modelo aquecido com sucesso.")
    except Exception as e:
        print(f"AVISO: Erro durante o aquecimento (pode ser normal): {e}")

    # 3. Coleta as respostas do sistema RAG para cada pergunta
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

    # 4. Prepara e executa a avaliação com RAGAs usando um LLM mais leve
    print("\nPreparando dados e executando a avaliação com RAGAs...")
    dataset = Dataset.from_list(results_for_ragas)
    
    # --- INÍCIO DA CORREÇÃO FINAL ---
    ragas_result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=get_ragas_llm(),  # <-- USA O "LLM JUIZ" (llama3:8b), MAIS RÁPIDO E OBEDIENTE
        embeddings=get_embedding_model_for_processing(), # Usa o embedding local que cria o índice
        raise_exceptions=False
    )
    # --- FIM DA CORREÇÃO FINAL ---
    
    print("Avaliação RAGAs concluída.")
    ragas_df = ragas_result.to_pandas()
    print("\n--- Resultados Médios da Avaliação RAGAs ---")
    print(ragas_df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean(skipna=True).round(3))
    print(f"\nSalvando resultados detalhados em '{RESULTS_CSV_PATH}'...")
    ragas_df.to_csv(RESULTS_CSV_PATH, index=False, encoding='utf-8-sig')
    print("Resultados salvos com sucesso.")