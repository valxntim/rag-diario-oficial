# src/rag_chain_builder.py (Versão Simplificada - SEM Re-ranker)

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# O Prompt continua o mesmo, pois as regras para o LLM são importantes desde o início
PROMPT_TEMPLATE_TEXT = """
Você é um assistente de IA especialista em analisar extratos de contratos do Diário Oficial. Sua única tarefa é extrair informações precisas do texto de contexto fornecido para responder à pergunta do usuário.

Regras:
1.  Use APENAS o texto fornecido na seção 'Contexto'. Não use nenhum conhecimento prévio.
2.  Se a pergunta pedir um valor monetário específico, sua resposta deve ser APENAS o valor (ex: "R$ 286.696,80"). Não adicione frases extras.
3.  Se a pergunta pedir uma informação textual (ex: nome da empresa, objeto), responda de forma concisa e direta.
4.  Se a informação exata para responder à pergunta não estiver no contexto, responda exatamente: "Informação não disponível no contexto."

Contexto:
{context}

Pergunta: {question}

Resposta:
"""

_cached_rag_chain = None

def build_rag_chain(llm, vector_store, force_reload=False):
    """
    Constrói a cadeia RAG mais simples possível, sem Re-ranker ou busca híbrida.
    Usa diretamente o retriever do vector store.
    """
    global _cached_rag_chain
    if _cached_rag_chain is not None and not force_reload:
        return _cached_rag_chain

    if not llm or not vector_store:
        print("ERRO: LLM ou Vector Store não fornecidos para build_rag_chain.")
        return None

    print("Construindo cadeia RAG SIMPLES (sem Re-ranker)...")
    try:
        # --- ETAPA 1: Criar o Retriever Básico ---
        # Ele vai simplesmente buscar os 'k' documentos mais similares do FAISS.
        # O valor de 'k' é definido quando o vector_store é criado ou usado.
        # No nosso caso, o config.py controla isso com RETRIEVER_SEARCH_K.
        from .config import RETRIEVER_SEARCH_K
        retriever = vector_store.as_retriever(
            search_kwargs={"k": RETRIEVER_SEARCH_K}
        )
        print(f"Retriever básico configurado com sucesso para buscar os top {RETRIEVER_SEARCH_K} documentos.")

        # --- ETAPA 2: Montar a Cadeia RetrievalQA ---
        prompt_for_answer = PromptTemplate(
            template=PROMPT_TEMPLATE_TEXT, input_variables=["context", "question"]
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever, # Usando o retriever simples
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_for_answer}
        )
        print("Cadeia RAG SIMPLES construída com sucesso.")
        
        _cached_rag_chain = qa_chain
        return qa_chain
        
    except Exception as e:
        print(f"ERRO ao construir a cadeia RAG simples: {e}")
        return None