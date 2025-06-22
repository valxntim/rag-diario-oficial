# src/llm_interface.py
"""
Interface para configurar e acessar os modelos Ollama, conectando-se a diferentes
instâncias para embeddings (local) e geração (remoto).
"""
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Importa as novas variáveis de configuração detalhadas
# LINHA CORRIGIDA
from .config import (
    OLLAMA_EMBEDDING_URL,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_LLM_URL,
    OLLAMA_LLM_MODEL
)

_cached_embeddings = None
_cached_llm = None

def get_ollama_embeddings(force_reload=False):
    """
    Inicializa e retorna o modelo de embedding Ollama, conectando-se
    à instância LOCAL especificada no config.
    """
    global _cached_embeddings
    if _cached_embeddings is not None and not force_reload:
        return _cached_embeddings

    print(f"Inicializando modelo de embedding: {OLLAMA_EMBEDDING_MODEL} via {OLLAMA_EMBEDDING_URL}")
    try:
        embeddings = OllamaEmbeddings(
            base_url=OLLAMA_EMBEDDING_URL,  # <-- USA A URL DE EMBEDDING
            model=OLLAMA_EMBEDDING_MODEL
        )
        _ = embeddings.embed_query("Teste de embedding inicial.")
        print(f"Modelo de embedding '{OLLAMA_EMBEDDING_MODEL}' carregado com sucesso.")
        _cached_embeddings = embeddings
        return embeddings
    except Exception as e:
        print(f"ERRO CRÍTICO ao inicializar OllamaEmbeddings para '{OLLAMA_EMBEDDING_MODEL}': {e}")
        raise

def get_ollama_llm(force_reload=False):
    """
    Inicializa e retorna o modelo de linguagem de geração Ollama, conectando-se
    à instância REMOTA especificada no config.
    """
    global _cached_llm
    if _cached_llm is not None and not force_reload:
        return _cached_llm

    print(f"Inicializando LLM de geração: {OLLAMA_LLM_MODEL} via {OLLAMA_LLM_URL}")
    try:
        llm = OllamaLLM(
            base_url=OLLAMA_LLM_URL,  # <-- USA A URL DO LLM
            model=OLLAMA_LLM_MODEL,
            temperature=0  # Garante respostas factuais e consistentes
        )
        _ = llm.invoke("Olá! Teste de LLM. Responda 'ok'.")
        print(f"LLM '{OLLAMA_LLM_MODEL}' carregado com sucesso.")
        _cached_llm = llm
        return llm
    except Exception as e:
        print(f"ERRO CRÍTICO ao inicializar OllamaLLM para '{OLLAMA_LLM_MODEL}': {e}")
        raise