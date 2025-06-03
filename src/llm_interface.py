# src/llm_interface.py
"""
Interface para configurar e acessar os modelos Ollama (Embeddings e LLM de Geração).
"""
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from .config import OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL, OLLAMA_LLM_MODEL

_cached_embeddings = None
_cached_llm = None

def get_ollama_embeddings(force_reload=False):
    """
    Inicializa e retorna o modelo de embedding Ollama. Cacheia a instância.
    """
    global _cached_embeddings
    if _cached_embeddings is not None and not force_reload:
        # print("Usando modelo de embedding Ollama do cache.") # Opcional
        return _cached_embeddings

    print(f"Inicializando modelo de embedding: {OLLAMA_EMBEDDING_MODEL} via {OLLAMA_BASE_URL}")
    try:
        embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_EMBEDDING_MODEL
        )
        # Teste rápido para garantir que está funcionando
        _ = embeddings.embed_query("Teste de embedding inicial.")
        print(f"Modelo de embedding '{OLLAMA_EMBEDDING_MODEL}' carregado com sucesso.")
        _cached_embeddings = embeddings
        return embeddings
    except Exception as e:
        print(f"ERRO CRÍTICO ao inicializar OllamaEmbeddings para '{OLLAMA_EMBEDDING_MODEL}': {e}")
        raise

def get_ollama_llm(force_reload=False):
    """
    Inicializa e retorna o modelo de linguagem de geração Ollama. Cacheia a instância.
    """
    global _cached_llm
    if _cached_llm is not None and not force_reload:
        # print("Usando LLM Ollama do cache.") # Opcional
        return _cached_llm

    print(f"Inicializando LLM de geração: {OLLAMA_LLM_MODEL} via {OLLAMA_BASE_URL}")
    try:
        llm = OllamaLLM(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_LLM_MODEL
        )
        # Teste rápido para garantir que está funcionando
        _ = llm.invoke("Olá! Teste de LLM. Responda 'ok'.") # Um invoke simples
        print(f"LLM '{OLLAMA_LLM_MODEL}' carregado com sucesso.")
        _cached_llm = llm
        return llm
    except Exception as e:
        print(f"ERRO CRÍTICO ao inicializar OllamaLLM para '{OLLAMA_LLM_MODEL}': {e}")
        raise

