# src/rag_chain_builder.py (Versão Simplificada - SEM Re-ranker)

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Em src/rag_chain_builder.py

PROMPT_TEMPLATE_TEXT = """
Você é um assistente de IA especialista e meticuloso, encarregado de analisar extratos do Diário Oficial. Sua tarefa é responder à pergunta do usuário de forma completa e precisa, utilizando exclusivamente o texto fornecido na seção 'Contexto'.

**Instruções Fundamentais:**
1.  **Extração Completa:** Sua resposta deve ser abrangente. Se a pergunta pede uma lista de critérios, forneça todos os critérios listados no contexto. Se pede um valor, forneça o valor e a descrição a que ele se refere. Extraia toda a informação relevante que responde diretamente à pergunta.
2.  **Fidelidade Absoluta ao Contexto:** Responda APENAS com base na informação presente no texto do 'Contexto'. Não presuma, infira ou utilize qualquer conhecimento externo.
3.  **Resposta Direta para Dados Específicos:** Se a pergunta for sobre um valor monetário, data, ou número de processo específico, e a resposta estiver claramente no texto, responda apenas com o dado extraído (ex: "R$ 286.696,80" ou "10/01/2019").
4.  **Recusa Honesta:** Se a informação necessária para responder à pergunta não estiver inequivocamente presente no contexto, responda exatamente: "A informação solicitada não foi encontrada no contexto fornecido."

**Contexto:**
{context}

**Pergunta:** {question}

**Resposta Precisa e Completa:**
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