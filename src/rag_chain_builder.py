# src/rag_chain_builder.py
"""
Constrói e configura a cadeia RAG (Retrieval Augmented Generation).
"""
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from .config import RETRIEVER_SEARCH_K

PROMPT_TEMPLATE_TEXT = """
Por favor, use APENAS os seguintes trechos de contexto para responder à pergunta no final.
Se a informação necessária não estiver presente no contexto fornecido, diga explicitamente "A informação não foi encontrada no contexto fornecido."
Não tente inventar uma resposta. Seja conciso e direto ao ponto. Forneça a resposta mais completa possível com base no contexto.

Contexto:
{context}

Pergunta: {question}

Resposta útil e concisa:
"""

_cached_rag_chain = None
_cached_llm_for_chain = None
_cached_vector_store_for_chain = None


def build_rag_chain(llm, vector_store, force_reload=False):
    """
    Constrói e retorna uma cadeia RetrievalQA configurada. Cacheia a instância.
    """
    global _cached_rag_chain, _cached_llm_for_chain, _cached_vector_store_for_chain
    
    if _cached_rag_chain is not None and \
       _cached_llm_for_chain == llm and \
       _cached_vector_store_for_chain == vector_store and \
       not force_reload:
        # print("Usando cadeia RAG do cache.") # Opcional
        return _cached_rag_chain

    if not llm or not vector_store:
        print("ERRO: LLM ou Vector Store não fornecidos para build_rag_chain.")
        return None

    print(f"Construindo nova cadeia RAG (k={RETRIEVER_SEARCH_K})...")
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_SEARCH_K})

        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE_TEXT, input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        print("Cadeia RAG (RetrievalQA) construída com sucesso.")
        _cached_rag_chain = qa_chain
        _cached_llm_for_chain = llm
        _cached_vector_store_for_chain = vector_store
        return qa_chain
    except Exception as e:
        print(f"ERRO ao construir a cadeia RAG: {e}")
        return None

