"""
INTERFACE DE MODELOS DE IA (LLM E EMBEDDINGS)
================================================
Gerencia a inicialização e cache de modelos de linguagem e embeddings.
Suporta tanto Ollama (servidor local/remoto) quanto HuggingFace (local).
Implementa cache de modelos para evitar recarregar a cada chamada.
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

# HuggingFace - biblioteca para usar modelos localmente
try:
    from sentence_transformers import SentenceTransformer
    _hf_available = True
except ImportError:
    _hf_available = False


# ========== CACHE GLOBAL ==========
# Armazena modelos já carregados para evitar recarregar
_cached_embeddings = None       # Cache para Ollama embeddings
_cached_llm = None              # Cache para LLM Ollama
_cached_hf_embeddings = None    # Cache para HuggingFace embeddings


def get_embeddings(force_reload=False):
    """
    Retorna o modelo de embedding conforme configurado em EMBEDDING_BACKEND.
    
    Argumentos:
        force_reload (bool): Se True, recarrega o modelo mesmo que esteja em cache
    
    Retorna:
        Objeto com métodos .embed_query() e .embed_documents() para gerar embeddings
    
    Levanta:
        ImportError: Se HuggingFace backend está configurado mas a biblioteca não está instalada
        ValueError: Se backend desconhecido foi configurado
    """
    global _cached_embeddings, _cached_hf_embeddings

    # ========== BACKEND OLLAMA ==========
    if EMBEDDING_BACKEND == "ollama":
        # Se já tem em cache e não quer recarregar, retorna o cache
        if _cached_embeddings is not None and not force_reload:
            return _cached_embeddings
        
        # Inicializa novo modelo Ollama para embeddings
        print(f"[EMB] Inicializando modelo Ollama: {OLLAMA_EMBEDDING_MODEL} via {OLLAMA_EMBEDDING_URL}")
        try:
            # Cria conexão com servidor Ollama
            embeddings = OllamaEmbeddings(
                base_url=OLLAMA_EMBEDDING_URL,
                model=OLLAMA_EMBEDDING_MODEL
            )
            # Testa se o modelo responde com um embedding teste
            _ = embeddings.embed_query("Teste de embedding inicial.")
            _cached_embeddings = embeddings
            print(f"[EMB] Modelo '{OLLAMA_EMBEDDING_MODEL}' carregado com sucesso.")
            return embeddings
        except Exception as e:
            print(f"ERRO CRÍTICO OllamaEmbeddings: {e}")
            raise
    
    # ========== BACKEND HUGGINGFACE ==========
    elif EMBEDDING_BACKEND == "huggingface":
        # Verifica se a biblioteca SentenceTransformer está instalada
        if not _hf_available:
            raise ImportError("sentence-transformers não está instalado. Execute: pip install sentence-transformers")
        
        # Se já tem em cache e não quer recarregar, retorna o cache
        if _cached_hf_embeddings is not None and not force_reload:
            return _cached_hf_embeddings
        
        # Inicializa novo modelo HuggingFace para embeddings
        print(f"[EMB] Inicializando HuggingFace embedding: {HF_EMBEDDING_MODEL}")
        try:
            # Baixa/carrega o modelo do HuggingFace
            model = SentenceTransformer(HF_EMBEDDING_MODEL)
            
            # Wrapper para compatibilidade com LangChain
            # LangChain espera métodos específicos: embed_query() e embed_documents()
            class HuggingFaceEmbedWrapper:
                """
                Wrapper que adapta SentenceTransformer para a interface LangChain.
                """
                def __init__(self, model):
                    self.model = model
                
                def embed_query(self, text):
                    """Gera embedding para uma única query (texto)"""
                    return self.model.encode([text])[0].tolist()
                
                def embed_documents(self, texts):
                    """Gera embeddings para múltiplos documentos (lista de textos)"""
                    return self.model.encode(texts).tolist()
                
                def __call__(self, texts):
                    """Permite chamar como função: embedder(text ou textos)"""
                    if isinstance(texts, str):
                        return self.embed_query(texts)
                    elif isinstance(texts, list):
                        return self.embed_documents(texts)
                    else:
                        raise TypeError("Input deve ser string ou lista de strings para embedding.")
            
            wrapper = HuggingFaceEmbedWrapper(model)
            # Testa se o wrapper responde corretamente
            _ = wrapper.embed_query("Teste de embedding inicial.")
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
    Inicializa e retorna o modelo LLM (geração de texto) via Ollama.
    
    Argumentos:
        force_reload (bool): Se True, recarrega o modelo mesmo que esteja em cache
    
    Retorna:
        Objeto OllamaLLM que pode gerar texto com .invoke(prompt)
    
    Levanta:
        Exception: Se não conseguir conectar ao servidor Ollama
    """
    global _cached_llm
    
    # Se já tem em cache e não quer recarregar, retorna o cache
    if _cached_llm is not None and not force_reload:
        return _cached_llm

    # Inicializa novo modelo Ollama para geração
    print(f"[LLM] Inicializando LLM: {OLLAMA_LLM_MODEL} via {OLLAMA_LLM_URL}")
    try:
        # Cria conexão com servidor Ollama
        llm = OllamaLLM(
            base_url=OLLAMA_LLM_URL,
            model=OLLAMA_LLM_MODEL,
            num_ctx=10000,           # Tamanho máximo de contexto que o modelo pode processar
            temperature=0            # 0 = respostas determinísticas, sem aleatoriedade
        )
        # Testa se o modelo responde
        _ = llm.invoke("Olá! Teste de LLM. Responda 'ok'.")
        _cached_llm = llm
        print(f"[LLM] Modelo '{OLLAMA_LLM_MODEL}' carregado com sucesso.")
        return llm
    except Exception as e:
        print(f"ERRO CRÍTICO OllamaLLM: {e}")
        raise