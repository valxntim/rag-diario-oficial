import os

# ===========================
# --- Caminhos Base ---
# ===========================

# Diretório do arquivo config.py (SRC_ROOT) e do projeto (PROJECT_ROOT)
SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_ROOT)

# ===========================
# --- Diretórios de Dados ---
# ===========================

# Pasta principal de dados
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# Pasta onde encontrar seus PDFs originais para indexar
PDF_DIRECTORY = os.path.join(DATA_DIR, "pdfs", "contratos_controlado")

# ===========================
# --- Dataset de Avaliação ---
# ===========================

# Caminho para o JSONL com perguntas e respostas de referência
DATASET_FILE_PATH = os.path.join(DATA_DIR, "benchmark_final_valor.jsonl")

# ===========================
# --- Configuração do Vector Store (FAISS) ---
# ===========================

# Diretório e nome do índice FAISS
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")
FAISS_INDEX_NAME = "faiss_index_chunk600_base_controlada"
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, FAISS_INDEX_NAME)

# ===========================
# --- Configurações dos Modelos de IA (Ollama) ---
# ===========================

# Parâmetros do LLM (geração, ex: Llama 3/4 via Ollama)
#OLLAMA_LLM_URL = "http://localhost:11434"    # Mude para o IP/porta de outro servidor se necessário
#OLLAMA_LLM_MODEL = "llama3.1:8b-32k"           # Altere conforme o modelo local disponível (veja com 'ollama list')



# Parâmetros do modelo de embedding via Ollama (caso use Ollama para embeddings)
#OLLAMA_EMBEDDING_URL = "http://localhost:11434"  # MESMO endereço do Ollama rodando modelo para embeddings
#OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large:latest"   # Ou outro, ex: "nomic-embed-text:latest"

# ===========================
# --- Parâmetros do Pipeline RAG ---
# ===========================

CHUNK_SIZE = 600            # Tamanho do chunk de texto para indexação
CHUNK_OVERLAP = 0        # Sobreposição dos chunks
RETRIEVER_SEARCH_K = 5    # Top K documentos mais similares a serem recuperados por consulta
#NUM_QUESTIONS_TO_TEST = 8 # Limita quantidade de perguntas em avaliações automáticas (ajuste se quiser debugar rápido)

# ===========================
# --- Backend de Embeddings ---
# ===========================

# Escolha entre "ollama" para usar embeddings servidos via Ollama, ou "huggingface" para usar modelos HuggingFace diretamente em Python
# Exemplo para HuggingFace (BERT jurídico):
#EMBEDDING_BACKEND = "huggingface"
HF_EMBEDDING_MODEL = "ulysses-camara/legal-bert-pt-br"

# Exemplo para Ollama (descomente, se for servir embeddings pelo Ollama):
EMBEDDING_BACKEND = "ollama"
#OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large:latest"
OLLAMA_EMBEDDING_MODEL = "bge-m3:latest"
OLLAMA_EMBEDDING_URL = "http://localhost:11434"

# ========== FIM (modifique apenas acima, se possível) ==========


#FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, FAISS_INDEX_NAME)

# --- Configurações dos Modelos de IA (Ollama) ---
#OLLAMA_LLM_URL = "http://localhost:11434"
#OLLAMA_LLM_MODEL = "llama3.1:8b-32k"
OLLAMA_LLM_URL = "http://164.41.75.221:11434"
OLLAMA_LLM_MODEL = "llama4:latest"       # Novo tag criado via Modelfile
# Em src/llm_interface.py ou onde você define o modelo
#OLLAMA_LLM_MODEL = "llama3.1:8b-32k"  # Mude de llama3.1:8b-32k
#OLLAMA_LLM_MODEL = "llama3.1:8b-32k"
#OLLAMA_LLM_MODEL = "llama3.1:8b-32k"
# --- MUDANÇA 1: Aponte para o novo modelo de embedding ---
# URL para o modelo de Embedding (agora usando o remoto também)
#OLLAMA_EMBEDDING_URL = "http://164.41.75.221:11434" 
#OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"  # <-- NOSSO NOVO CANDIDATO



#OLLAMA_EMBEDDING_URL = "http://localhost:11434"
#OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large:latest"
#OLLAMA_EMBEDDING_MODEL = "paraphrase-multilingual:278m-mpnet-base-v2-fp16" 

#EMBEDDING_BACKEND = "ollama"
#OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large:latest"
#OLLAMA_EMBEDDING_URL = "http://localhost:11434"
