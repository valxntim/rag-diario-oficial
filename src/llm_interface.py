# src/llm_interface.py
"""
Interface para configurar e acessar os modelos do projeto.
Inclui modelos de geração (Ollama) e modelos de embedding (Hugging Face).
"""
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings # <-- NOVO IMPORT

# Import do seu arquivo de configuração
from .config import (
    OLLAMA_EMBEDDING_URL,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_LLM_URL,
    OLLAMA_LLM_MODEL
)

# --- MODELOS DE EMBEDDING ---

_cached_ollama_embeddings = None
_cached_specialized_embeddings = None

def get_ollama_embeddings(force_reload=False):
    """
    Inicializa um modelo de embedding genérico via Ollama.
    """
    global _cached_ollama_embeddings
    if _cached_ollama_embeddings is not None and not force_reload:
        return _cached_ollama_embeddings
    # ... (o resto da sua função get_ollama_embeddings pode continuar aqui se quiser mantê-la para testes)
    return None # Desativado por padrão para focar no especializado

# --- NOVA FUNÇÃO DE EMBEDDING ESPECIALIZADO ---
def get_specialized_embeddings(force_reload=False):
    """
    Carrega o modelo de embedding especializado que será usado para
    segmentação e para o Vector Store.
    """
    global _cached_specialized_embeddings
    if _cached_specialized_embeddings is not None and not force_reload:
        return _cached_specialized_embeddings

    # RECOMENDAÇÃO FORTE: Use o modelo treinado para o domínio jurídico.
    # Ele entenderá melhor o "juridiquês" e os erros de OCR.
    # model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Genérico
    model_name = "rufimelo/Legal-BERTimbau-sts-large" # Especializado

    print(f"Inicializando embedding model especializado: {model_name}")
    
    # Usamos o wrapper do LangChain que é compatível com tudo.
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},  # Mude para 'cuda' se tiver GPU
        encode_kwargs={'normalize_embeddings': True} # Bom para similaridade de cosseno
    )
    
    _cached_specialized_embeddings = embeddings
    print(f"Modelo de embedding '{model_name}' carregado com sucesso.")
    return embeddings

# --- MODELO DE GERAÇÃO (LLM) ---

_cached_llm = None
def get_ollama_llm(force_reload=False):
    """
    Inicializa e retorna o modelo de linguagem de geração Ollama.
    (Esta função permanece a mesma).
    """
    global _cached_llm
    if _cached_llm is not None and not force_reload:
        return _cached_llm

    print(f"Inicializando LLM de geração: {OLLAMA_LLM_MODEL} via {OLLAMA_LLM_URL}")
    try:
        llm = OllamaLLM(
            base_url=OLLAMA_LLM_URL,
            model=OLLAMA_LLM_MODEL,
            temperature=0,
            timeout=300 # Mantendo o timeout que adicionamos
        )
        _ = llm.invoke("Olá!")
        print(f"LLM '{OLLAMA_LLM_MODEL}' carregado com sucesso.")
        _cached_llm = llm
        return llm
    except Exception as e:
        print(f"ERRO CRÍTICO ao inicializar OllamaLLM para '{OLLAMA_LLM_MODEL}': {e}")
        raise