"""
Interface para configurar e acessar os modelos de linguagem e embeddings,
suportando tanto Ollama quanto HuggingFace SentenceTransformer.
"""

from .config import (
    OLLAMA_EMBEDDING_URL,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_LLM_URL,
    OLLAMA_LLM_MODEL,
    EMBEDDING_BACKEND,
    HF_EMBEDDING_MODEL,
)

from langchain_ollama import OllamaEmbeddings, OllamaLLM

# HuggingFace
try:
    from sentence_transformers import SentenceTransformer
    _hf_available = True
except ImportError:
    _hf_available = False

_cached_embeddings = None
_cached_llm = None
_cached_hf_embeddings = None

def get_embeddings(force_reload=False):
    """
    Retorna o modelo de embedding de acordo com backend selecionado no config.
    """
    global _cached_embeddings, _cached_hf_embeddings

    if EMBEDDING_BACKEND == "ollama":
        if _cached_embeddings is not None and not force_reload:
            return _cached_embeddings
        print(f"[EMB] Inicializando modelo Ollama: {OLLAMA_EMBEDDING_MODEL} via {OLLAMA_EMBEDDING_URL}")
        try:
            embeddings = OllamaEmbeddings(
                base_url=OLLAMA_EMBEDDING_URL,
                model=OLLAMA_EMBEDDING_MODEL
            )
            _ = embeddings.embed_query("Teste de embedding inicial.")
            _cached_embeddings = embeddings
            print(f"[EMB] Modelo '{OLLAMA_EMBEDDING_MODEL}' carregado com sucesso.")
            return embeddings
        except Exception as e:
            print(f"ERRO CRÍTICO OllamaEmbeddings: {e}")
            raise
    elif EMBEDDING_BACKEND == "huggingface":
        if not _hf_available:
            raise ImportError("sentence-transformers não está instalado.")
        if _cached_hf_embeddings is not None and not force_reload:
            return _cached_hf_embeddings
        print(f"[EMB] Inicializando HuggingFace embedding: {HF_EMBEDDING_MODEL}")
        try:
            model = SentenceTransformer(HF_EMBEDDING_MODEL)
            # ---- Wrapper compatível com LangChain (.embed_query, .embed_documents, __call__) ----
            class HuggingFaceEmbedWrapper:
                def __init__(self, model):
                    self.model = model
                def embed_query(self, text):
                    return self.model.encode([text])[0].tolist()
                def embed_documents(self, texts):
                    return self.model.encode(texts).tolist()
                def __call__(self, texts):
                    if isinstance(texts, str):
                        return self.embed_query(texts)
                    elif isinstance(texts, list):
                        return self.embed_documents(texts)
                    else:
                        raise TypeError("Input must be a string or list of strings for embedding.")
            wrapper = HuggingFaceEmbedWrapper(model)
            _ = wrapper.embed_query("Teste de embedding inicial.")  # sanity check
            _cached_hf_embeddings = wrapper
            print(f"[EMB] Modelo '{HF_EMBEDDING_MODEL}' carregado com sucesso.")
            return wrapper
        except Exception as e:
            print(f"ERRO CRÍTICO HuggingFace embeddings: {e}")
            raise
    else:
        raise ValueError(f"Embedding backend desconhecido: '{EMBEDDING_BACKEND}'")

def get_llm(force_reload=False):
    """
    Inicializa e retorna o modelo de linguagem de geração Ollama, conectando-se
    à instância REMOTA especificada no config.
    (Você pode adicionar HuggingFace LLMs futuramente!)
    """
    global _cached_llm
    if _cached_llm is not None and not force_reload:
        return _cached_llm

    print(f"[LLM] Inicializando LLM: {OLLAMA_LLM_MODEL} via {OLLAMA_LLM_URL}")
    try:
        llm = OllamaLLM(
            base_url=OLLAMA_LLM_URL,
            model=OLLAMA_LLM_MODEL,
            temperature=0
        )
        _ = llm.invoke("Olá! Teste de LLM. Responda 'ok'.")
        _cached_llm = llm
        print(f"[LLM] Modelo '{OLLAMA_LLM_MODEL}' carregado com sucesso.")
        return llm
    except Exception as e:
        print(f"ERRO CRÍTICO OllamaLLM: {e}")
        raise
