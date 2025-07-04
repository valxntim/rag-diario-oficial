# src/config.py (Versão Corrigida para a Análise do "Fusca")

import os

# --- Caminhos Base ---
SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_ROOT)

# --- Diretórios de Dados ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PDF_DIRECTORY = os.path.join(DATA_DIR, "pdfs", "contratos")

# --- Configuração do Vector Store (Índice FAISS) ---
# Nome descritivo para o índice desta fase de testes
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")
#FAISS_INDEX_NAME = "faiss_index_baseline_recursive_750_150" # Nome claro com os parâmetros


FAISS_INDEX_NAME = "faiss_index_chunk256_overlap64" # O nome reflete os parâmetros!
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, FAISS_INDEX_NAME)

# --- Configurações dos Modelos de IA (Ollama) ---

# URL para o modelo de Geração (servidor remoto)
OLLAMA_LLM_URL = "http://164.41.75.221:11434"
OLLAMA_LLM_MODEL = "llama4:latest"

# URL para o modelo de Embedding (sua máquina local)
# Lembrete: Seu serviço Ollama local precisa estar rodando para que isso funcione!
OLLAMA_EMBEDDING_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large:latest"

# --- Parâmetros de Teste para o RAG "Fusca" ---


# Tamanho dos chunks e sobreposição. Valores menores criam chunks mais focados.
# Exemplo para o Teste 1.1
CHUNK_SIZE = 256
CHUNK_OVERLAP = 64
RETRIEVER_SEARCH_K = 15