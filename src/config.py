# src/config.py
"""
Arquivo de Configurações Globais para o Projeto RAG.
"""
import os

# --- Caminhos Base ---
SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_ROOT)

# --- Diretórios de Dados ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Em src/config.py
PDF_DIRECTORY = os.path.join(DATA_DIR, "pdfs", "contratos") # Verifique este caminho
# Mude o nome do índice para não usar o antigo




VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")
# NOVO NOME para o índice FAISS para garantir que não usamos o antigo.
#FAISS_INDEX_NAME = "faiss_index_contratos_final_ollama"
#FAISS_INDEX_NAME = "faiss_index_documento_inteiro"
#FAISS_INDEX_NAME = "faiss_index_do_dataset_enriquecido"
# Altere esta linha no seu config.py
FAISS_INDEX_NAME = "faiss_index_new"
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, FAISS_INDEX_NAME)

# URL para o modelo de Geração (servidor remoto)
#OLLAMA_LLM_URL = "http://164.41.75.221:11434"
#OLLAMA_LLM_MODEL = "llama4:latest"
OLLAMA_LLM_URL = "http://localhost:11434"  # URL local
OLLAMA_LLM_MODEL = "deepseek-r1:8b"  # Modelo local


# URL para o modelo de Embedding (sua máquina local)
# O Ollama na sua máquina local geralmente roda em 'http://localhost:11434'
OLLAMA_EMBEDDING_URL = "http://localhost:11434" 
OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large:latest"

# RETRIEVER_SEARCH_K deve continuar em 1 por enquanto
RETRIEVER_SEARCH_K = 20


CHUNK_SIZE = 400
CHUNK_OVERLAP = 100


# --- Perguntas de Exemplo para o Chatbot ---
EXAMPLE_QUESTIONS = [
    "Qual o nome fantasia da empresa que apresentou a proposta de cooperação à Administração Regional do Guará mencionada na Ordem de Serviço Nº 54 de 2021?",
    "Quem deverá apresentar o projeto de viabilidade simplificado?"
]