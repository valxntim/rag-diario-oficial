# src/config.py
"""
Arquivo de Configurações Globais para o Projeto RAG.
"""
import os
# Em src/config.py
DEVICE = "cuda" # Ou "cpu", dependendo do seu hardware
# --- Caminhos Base ---
SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_ROOT)

# --- Diretórios de Dados ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PDF_DIRECTORY = os.path.join(DATA_DIR, "pdfs", "contratos") # Verifique se este é o caminho correto para os seus PDFs

# --- Configuração do Vector Store (Índice FAISS) ---
# Usamos um nome de índice claro para o modelo que estamos usando.
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")
FAISS_INDEX_NAME = "faiss_index_nomic_embed"
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, FAISS_INDEX_NAME)

# --- Configurações dos Modelos de IA (Ollama) ---

# Ambas as URLs agora apontam para o mesmo servidor remoto para consistência.
OLLAMA_BASE_URL = "http://164.41.75.221:11434"

# Modelo de Geração (LLM)
OLLAMA_LLM_MODEL = "llama4:latest"

# Modelo de Embedding (Especializado, que está no mesmo servidor)
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"

# --- Configurações do Retriever ---
# Número de documentos a serem recuperados antes do re-ranking.
RETRIEVER_SEARCH_K = 20

# Configurações de como o texto é dividido (opcional, pode ajustar depois)
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

# --- Perguntas de Exemplo para o Chatbot ---
EXAMPLE_QUESTIONS = [
    "Qual o nome fantasia da empresa que apresentou a proposta de cooperação à Administração Regional do Guará mencionada na Ordem de Serviço Nº 54 de 2021?",
    "Quem deverá apresentar o projeto de viabilidade simplificado?"
]