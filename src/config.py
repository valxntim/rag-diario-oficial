"""
CONFIGURAÇÃO CENTRAL DO SISTEMA RAG
======================================
Este arquivo centraliza TODAS as configurações do projeto RAG para Diário Oficial.
Modifique apenas as variáveis abaixo conforme seu ambiente/necessidade.
"""

import os


# ===========================
# --- CAMINHOS BASE ---
# ===========================
# Define os diretórios raiz do projeto para referenciar recursos

# Diretório onde este arquivo (config.py) está localizado
SRC_ROOT = os.path.dirname(os.path.abspath(__file__))

# Diretório pai de src/ (raiz do projeto)
PROJECT_ROOT = os.path.dirname(SRC_ROOT)


# ===========================
# --- DIRETÓRIOS DE DADOS ---
# ===========================
# Estrutura de pastas para armazenar dados, PDFs e índices vetoriais

# Pasta principal onde todos os dados serão armazenados ( Onde o FAIS vai ser guardado )
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Pasta onde você coloca os PDFs originais que serão indexados
# IMPORTANTE: Coloque seus PDFs em: data/pdfs/contratos_validado/
PDF_DIRECTORY = os.path.join(DATA_DIR, "pdfs", "contratos_validado")


# ===========================
# --- DATASET DE AVALIAÇÃO ---
# ===========================
# Arquivo JSONL com pares pergunta/resposta para benchmarking

# Caminho para o dataset de avaliação (formato JSONL: 1 JSON por linha)
# Cada linha deve conter: id_versao_pergunta, pergunta, resposta, pdf, extrato
DATASET_FILE_PATH = os.path.join(DATA_DIR, "benchmark_final_valor.jsonl")


# ===========================
# --- VECTOR STORE (FAISS) ---
# ===========================
# Configuração do índice FAISS para busca vetorial semântica

# Diretório onde o índice FAISS será salvo/carregado
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")

# Nome do índice FAISS (será sufixado com chunk_size)
# IMPORTANTE: O nome identifica a versão específica dos chunks
FAISS_INDEX_NAME = "faiss_index_chunk600_base_validado"

# Caminho completo do índice FAISS
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, FAISS_INDEX_NAME)


# ===========================
# --- MODELOS DE IA (OLLAMA) ---
# ===========================
# URLs e nomes dos modelos rodando via Ollama (ou servidor remoto)

# URL do servidor Ollama para geração de texto (LLM)
# Se está em outro servidor, mude para: http://IP_REMOTO:11434
OLLAMA_LLM_URL = "http://localhost:11434"

# Nome/tag do modelo LLM disponível no Ollama
# Execute: ollama list para ver modelos instalados
OLLAMA_LLM_MODEL = "llama3.1:8b-32k"


# ===========================
# --- PARÂMETROS DO PIPELINE RAG ---
# ===========================
# Configurações de chunking e retrieval

# Tamanho de cada chunk de texto (em caracteres)
# Recomendado: 400-1000. Maiores = mais contexto, menos chunks. Menores = mais específico.
CHUNK_SIZE = 600

# Sobreposição entre chunks consecutivos (em caracteres)
# Ajuda a evitar cortes no meio de frases. Típico: 0-100.
CHUNK_OVERLAP = 0

# Número de documentos mais similares a recuperar por query
# Quanto maior, mais contexto (e mais lento). Típico: 3-10
RETRIEVER_SEARCH_K = 7


# ===========================
# --- BACKEND DE EMBEDDINGS ---
# ===========================
# Escolha qual serviço gera os embeddings (vetores semânticos)

# Escolha entre "ollama" (servir via Ollama) ou "huggingface" (local em Python)
EMBEDDING_BACKEND = "ollama"

# --------- Se usar Ollama para embeddings ---------
OLLAMA_EMBEDDING_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "bge-m3:latest"

# --------- Se usar HuggingFace (comentado, descomente se quiser usar) ---------
HF_EMBEDDING_MODEL = "ulysses-camara/legal-bert-pt-br"  # Modelo jurídico em PT-BR

