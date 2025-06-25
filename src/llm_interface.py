# src/llm_interface.py
"""
Interface para configurar e acessar os modelos do projeto.
Inclui modelos de geração e embedding, ambos via Ollama.
"""
# 1. Import ATUALIZADO para usar a variável unificada
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from .config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_LLM_MODEL
)

# 2. Variáveis de cache (sem alteração)
_cached_specialized_embeddings = None
_cached_llm = None


# 3. Função de embedding usando a nova variável
def get_specialized_embeddings(force_reload=False):
    """
    Inicializa e retorna o modelo de embedding especializado via Ollama.
    Utiliza cache para evitar recarregamentos desnecessários.
    """
    global _cached_specialized_embeddings
    if _cached_specialized_embeddings is not None and not force_reload:
        return _cached_specialized_embeddings

    # Mensagem de log atualizada para usar a variável correta
    print(f"Inicializando embedding model via Ollama: {OLLAMA_EMBEDDING_MODEL} em {OLLAMA_BASE_URL}")
    try:
        embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,  # <-- Alterado para a URL base unificada
            model=OLLAMA_EMBEDDING_MODEL
        )
        # Pequeno teste para garantir que o modelo está acessível
        embeddings.embed_query("Teste de conexão.")
        print(f"Modelo de embedding '{OLLAMA_EMBEDDING_MODEL}' carregado com sucesso.")
        _cached_specialized_embeddings = embeddings
        return _cached_specialized_embeddings
    except Exception as e:
        print(f"ERRO CRÍTICO ao inicializar OllamaEmbeddings para '{OLLAMA_EMBEDDING_MODEL}': {e}")
        print(f"Verifique se o modelo foi baixado com 'OLLAMA_HOST={OLLAMA_BASE_URL} ollama pull {OLLAMA_EMBEDDING_MODEL}' e se o serviço Ollama está rodando.")
        raise


# 4. Função de Geração (LLM) usando a nova variável
def get_ollama_llm(force_reload=False):
    """
    Inicializa e retorna o modelo de linguagem de geração Ollama.
    """
    global _cached_llm
    if _cached_llm is not None and not force_reload:
        return _cached_llm

    # Mensagem de log atualizada para usar a variável correta
    print(f"Inicializando LLM de geração: {OLLAMA_LLM_MODEL} via {OLLAMA_BASE_URL}")
    try:
        llm = OllamaLLM(
            base_url=OLLAMA_BASE_URL, # <-- Alterado para a URL base unificada
            model=OLLAMA_LLM_MODEL,
            temperature=0,
            timeout=300
        )
        # Teste de invocação
        _ = llm.invoke("Olá!")
        print(f"LLM '{OLLAMA_LLM_MODEL}' carregado com sucesso.")
        _cached_llm = llm
        return llm
    except Exception as e:
        print(f"ERRO CRÍTICO ao inicializar OllamaLLM para '{OLLAMA_LLM_MODEL}': {e}")
        raise
# Adicione esta nova função em src/llm_interface.py

def get_ragas_llm(force_reload=False):
    """
    Inicializa um LLM mais leve e rápido, ideal para as tarefas de avaliação do RAGAs.
    """
    # Usaremos o llama3:8b como nosso "juiz"
    RAGAS_LLM_MODEL = "llama3:8b" 

    # Usamos a mesma URL base do config
    from .config import OLLAMA_BASE_URL

    print(f"Inicializando LLM de avaliação RAGAs: {RAGAS_LLM_MODEL} via {OLLAMA_BASE_URL}")
    try:
        # Damos um timeout bem generoso para evitar erros de conexão
        ragas_llm = OllamaLLM(
            base_url=OLLAMA_BASE_URL,
            model=RAGAS_LLM_MODEL,
            temperature=0,
            timeout=600 # Timeout de 10 minutos
        )
        _ = ragas_llm.invoke("Olá, você está pronto para avaliar?")
        print(f"LLM de avaliação '{RAGAS_LLM_MODEL}' carregado com sucesso.")
        return ragas_llm
    except Exception as e:
        print(f"ERRO CRÍTICO ao inicializar o LLM para RAGAs: {e}")
        raise