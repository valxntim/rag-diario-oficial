# converter_dataset.py
"""
Este script utilitário lê o dataset mestre em formato CSV (golden_dataset.csv)
e o converte para o formato JSONL (JSON Lines) que o script de avaliação
evaluate_rag.py espera.
"""
import csv
import json
import os

INPUT_CSV_FILE = "golden_dataset.csv"
OUTPUT_JSONL_FILE = "dataset_para_teste.jsonl"

def convert_csv_to_jsonl():
    """
    Lê o arquivo CSV e escreve um arquivo JSONL.
    """
    print(f"--- Iniciando conversão de '{INPUT_CSV_FILE}' para '{OUTPUT_JSONL_FILE}' ---")

    if not os.path.exists(INPUT_CSV_FILE):
        print(f"ERRO: Arquivo de entrada '{INPUT_CSV_FILE}' não encontrado. Por favor, crie-o primeiro.")
        return

    questions_data = []
    with open(INPUT_CSV_FILE, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # O script de avaliação espera as chaves 'question' e 'answer'
            # Estamos mapeando as colunas do seu CSV para essas chaves.
            questions_data.append({
                "id": row.get("question_id"),
                "question": row.get("question"),
                "answer": row.get("ground_truth") # Mapeando 'ground_truth' para 'answer'
            })

    with open(OUTPUT_JSONL_FILE, 'w', encoding='utf-8') as jsonlfile:
        for entry in questions_data:
            jsonlfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Conversão concluída com sucesso! {len(questions_data)} perguntas foram salvas em '{OUTPUT_JSONL_FILE}'.")
    print("Este arquivo agora pode ser usado pelo seu script 'evaluate_rag.py'.")

if __name__ == "__main__":
    convert_csv_to_jsonl()