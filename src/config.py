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
# ATENÇÃO: Ajuste para a pasta correta dos seus PDFs para operação normal
PDF_DIRECTORY = os.path.join(DATA_DIR, "pdfs", "pessoal") 
# PDF_DIRECTORY = os.path.join(DATA_DIR, "pdfs", "pdf_unico_teste_guara") # Exemplo para teste específico
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")
FAISS_INDEX_NAME = "faiss_index_diario_oficial_ollama"
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, FAISS_INDEX_NAME)

# --- Configurações do Ollama ---
OLLAMA_BASE_URL = "http://164.41.75.221:11434"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_LLM_MODEL = "llama4:latest" # Ou sua escolha (ex: "llama3:8b")

# --- Configurações de Processamento de Texto ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Configurações da Cadeia RAG ---
RETRIEVER_SEARCH_K = 5 # Número de chunks a serem recuperados (ajustado de 3 para 5)

# --- Perguntas de Exemplo para o Chatbot ---
EXAMPLE_QUESTIONS = [
    "Qual o nome fantasia da empresa que apresentou a proposta de cooperação à Administração Regional do Guará mencionada na Ordem de Serviço Nº 54 de 2021?",
    "Quem representa a empresa GUARÁ ECO?",
    "Qual o objeto da proposta de cooperação apresentada por GUARÁ ECO?",
    "Onde se localiza o mobiliário urbano que receberá as benfeitorias propostas por GUARÁ ECO?",
    "Qual o número do processo referente à proposta de cooperação da GUARÁ ECO na Ordem de Serviço Nº 54 de 2021?",
    "Qual o CNPJ da GUARÁ ECO?",
    "Quem deverá apresentar o projeto de viabilidade simplificado?" # Pergunta que funcionou
]

# Query para teste de recuperação (opcional, pode ser comentada em chatbot_cli.py)
QUERY_TEST_RETRIEVAL = "proposta de cooperação GUARÁ ECO"
# QUERY_TEST_RETRIEVAL = "Qual o valor do crédito suplementar?"

