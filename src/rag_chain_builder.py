# src/rag_chain_builder.py
"""
Constrói a cadeia RAG usando uma arquitetura avançada e moderna de duas etapas:
1. Retriever de busca ampla (Recall).
2. Re-ranker com Cross-Encoder para precisão (Precision).
"""
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

# --- INÍCIO DA CORREÇÃO DEFINITIVA ---
# Com base no diagnóstico, esta é a importação correta para a sua versão.
from langchain_community.cross_encoders.huggingface import HuggingFaceCrossEncoder
# --- FIM DA CORREÇÃO DEFINITIVA ---

from .config import RETRIEVER_SEARCH_K

# Template do prompt para a resposta final do LLM (permanece o mesmo)
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
    Constrói e retorna uma cadeia RetrievalQA configurada com um Re-ranker (Cross-Encoder).
    """
    global _cached_rag_chain
    if _cached_rag_chain is not None and not force_reload:
        return _cached_rag_chain

    if not llm or not vector_store:
        print("ERRO: LLM ou Vector Store não fornecidos para build_rag_chain.")
        return None

    # No config.py, lembre-se de usar um k maior, como 20.
    print(f"Construindo nova cadeia RAG com Cross-Encoder Re-ranker (busca inicial k={RETRIEVER_SEARCH_K})...")
    try:
        # --- ETAPA 1: Configurar o Retriever de Busca Ampla ---
        base_retriever = vector_store.as_retriever(
            search_kwargs={"k": RETRIEVER_SEARCH_K}
        )

        # --- ETAPA 2: Configurar o Modelo Cross-Encoder usando a classe correta ---
        # --- INÍCIO DA CORREÇÃO DEFINITIVA ---
        model = HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
        # --- FIM DA CORREÇÃO DEFINITIVA ---
        
        # --- ETAPA 3: Configurar o Compressor/Re-ranker ---
        # Agora o 'model' que passamos é do tipo que o CrossEncoderReranker espera.
        compressor = CrossEncoderReranker(model=model, top_n=3)
        
        # --- ETAPA 4: Criar o Retriever Final com Compressão ---
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
        print("Retriever com Re-ranker (HuggingFaceCrossEncoder) configurado com sucesso.")

        # --- ETAPA 5: Montar a Cadeia RetrievalQA (como antes) ---
        prompt_for_answer = PromptTemplate(
            template=PROMPT_TEMPLATE_TEXT, input_variables=["context", "question"]
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=compression_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_for_answer}
        )
        print("Cadeia RAG (RetrievalQA) com Re-ranker construída com sucesso.")
        
        _cached_rag_chain = qa_chain
        
        return qa_chain
    except Exception as e:
        print(f"ERRO ao construir a cadeia RAG: {e}")
        return None