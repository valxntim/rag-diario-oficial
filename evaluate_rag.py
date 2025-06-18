# evaluate_rag.py
"""
Script para avaliar automaticamente o sistema RAG usando um dataset JSONL.
"""
import sys
import os
import json
import re
import csv
import time
from tqdm import tqdm # Para uma barra de progresso visual

# --- CORREÇÃO IMPORTANTE AQUI ---
# Adiciona o diretório raiz do projeto ao Python Path.
# Isto garante que o Python consegue encontrar o pacote 'src' e seus módulos.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Agora as importações do 'src' devem funcionar
from src.llm_interface import get_ollama_llm
from src.vector_store_manager import get_vector_store
from src.rag_chain_builder import build_rag_chain

# --- Configurações da Avaliação ---
# Caminho para o seu arquivo de perguntas e respostas
DATASET_FILE_PATH = "./dataset_enriquecido.jsonl"
# Caminho para o arquivo de resultados
RESULTS_CSV_PATH = "./evaluation_results_qa.csv"
# Número de perguntas a serem testadas
NUM_QUESTIONS_TO_TEST = 100

def extract_monetary_value(text: str) -> str or None:
    """
    Extrai e normaliza um valor monetário de uma string.
    """
    if not isinstance(text, str):
        return None
    match = re.search(r'R\$\s*([\d\.,]+)', text)
    if match:
        value_str = match.group(1)
        normalized_value = value_str.strip().replace('.', '').replace(',', '.')
        try:
            float(normalized_value)
            return normalized_value
        except ValueError:
            return None
    return None

def load_evaluation_data(file_path: str):
    """
    Carrega o dataset de avaliação a partir de um arquivo JSONL.
    """
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

def run_evaluation():
    """
    Função principal que executa a avaliação automatizada.
    """
    print("--- Iniciando Avaliação Automatizada do Sistema RAG ---")

    evaluation_data = load_evaluation_data(DATASET_FILE_PATH)
    if not evaluation_data:
        print("Nenhum dado encontrado para avaliação. Encerrando.")
        return

    if NUM_QUESTIONS_TO_TEST is not None:
        print(f"ATENÇÃO: Testando apenas as primeiras {NUM_QUESTIONS_TO_TEST} perguntas do dataset.")
        evaluation_data = evaluation_data[:NUM_QUESTIONS_TO_TEST]

    print("\nInicializando o sistema RAG...")
    try:
        # Para forçar a recriação do índice, use: get_vector_store(force_recreate=True)
        vector_store = get_vector_store()
        llm = get_ollama_llm()
        qa_chain = build_rag_chain(llm, vector_store)
        
        if not all([vector_store, llm, qa_chain]):
             raise Exception("Falha ao inicializar um ou mais componentes do RAG.")
             
        print("Sistema RAG inicializado com sucesso.")
    except Exception as e:
        print(f"ERRO CRÍTICO ao inicializar o sistema RAG: {e}")
        return

    results = []
    correct_answers = 0
    total_questions = len(evaluation_data)
    
    print(f"\nIniciando a avaliação de {total_questions} perguntas...")

    for item in tqdm(evaluation_data, desc="Avaliando perguntas"):
        question = item.get("question")
        ground_truth_answer = item.get("answer")

        if not question or not ground_truth_answer:
            continue

        try:
            result = qa_chain.invoke({"query": question})
            generated_answer = result.get("result", "")
        except Exception as e:
            generated_answer = f"ERRO NA EXECUÇÃO: {e}"

        expected_value = extract_monetary_value(ground_truth_answer)
        generated_value = extract_monetary_value(generated_answer)

        is_correct = False
        if expected_value is not None and generated_value is not None:
            if abs(float(expected_value) - float(generated_value)) < 0.01:
                is_correct = True
                correct_answers += 1
        
        results.append({
            "id": item.get("id"),
            "is_correct": is_correct,
            "question": question,
            "ground_truth_answer": ground_truth_answer,
            "generated_answer": generated_answer.strip().replace('\n', ' '),
            "expected_value": expected_value,
            "generated_value": generated_value,
            "source_documents_pages": [doc.metadata.get('page', -1) + 1 for doc in result.get("source_documents", []) if 'page' in doc.metadata]
        })

    accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    
    print("\n--- Resultados Finais da Avaliação ---")
    print(f"Total de Perguntas Avaliadas: {total_questions}")
    print(f"Respostas Corretas (Valor Exato): {correct_answers}")
    print(f"Acurácia (Exact Match de Valor): {accuracy:.2f}%")
    
    print(f"\nSalvando resultados detalhados em '{RESULTS_CSV_PATH}'...")
    try:
        with open(RESULTS_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                "id", "is_correct", "question", "ground_truth_answer", "generated_answer", 
                "expected_value", "generated_value", "source_documents_pages"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print("Resultados salvos com sucesso.")
    except Exception as e:
        print(f"ERRO ao salvar o arquivo CSV de resultados: {e}")

if __name__ == "__main__":
    run_evaluation()
