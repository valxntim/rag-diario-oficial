# inspect_chunks.py
"""
Este script serve como uma ferramenta de depuração para visualizar os chunks
gerados pelo 'vector_store_manager.py'. Ele não cria um índice, apenas
salva os chunks em um arquivo CSV para fácil inspeção humana.
"""
import sys
import os
import csv

# Adiciona o diretório raiz ao Python Path para encontrar o pacote 'src'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Importa a função de chunking e a configuração do diretório de PDFs
from src.vector_store_manager import load_and_chunk_pdfs
from src.config import PDF_DIRECTORY

# Nome do arquivo de saída onde os resultados serão salvos
OUTPUT_CSV_FILE = "inspecao_chunks.csv"

def inspect_and_save_chunks():
    """
    Função principal que executa a inspeção.
    """
    print("--- Iniciando Inspeção de Chunks ---")
    
    # 1. Usa a mesma função do seu sistema principal para criar os chunks
    # Isso garante que você está vendo exatamente o que seu RAG usaria.
    list_of_chunks = load_and_chunk_pdfs(PDF_DIRECTORY)
    
    if not list_of_chunks:
        print("Nenhum chunk foi criado. Encerrando a inspeção.")
        return

    print(f"\nTotal de {len(list_of_chunks)} chunks encontrados. Preparando para salvar em '{OUTPUT_CSV_FILE}'...")

    # 2. Salva os chunks em um arquivo CSV para análise
    try:
        with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            # Define o cabeçalho das colunas
            fieldnames = ["Numero_Chunk", "Pagina", "Fonte_PDF", "Conteudo_Chunk"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Escreve cada chunk em uma nova linha
            for i, chunk in enumerate(list_of_chunks):
                writer.writerow({
                    "Numero_Chunk": i + 1,
                    "Pagina": chunk.metadata.get('page', 'N/A'),
                    "Fonte_PDF": chunk.metadata.get('source', 'N/A'),
                    "Conteudo_Chunk": chunk.page_content
                })
        
        print(f"\nInspeção concluída com sucesso! Resultados salvos em '{OUTPUT_CSV_FILE}'.")
        print("Você pode abrir este arquivo no Excel, Google Sheets ou qualquer editor de planilhas.")

    except Exception as e:
        print(f"\nOcorreu um erro ao salvar o arquivo CSV: {e}")

if __name__ == "__main__":
    inspect_and_save_chunks()