from tqdm import tqdm
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

PROMPT_TEMPLATE_LEGAL = """
Você é um assistente especializado em encontrar valores monetários em contratos públicos do Diário Oficial brasileiro.

Instruções:
1. Analise cuidadosamente todos os contextos fornecidos.
2. Sua tarefa é identificar o valor monetário EXATO que responde à pergunta abaixo, considerando as informações sobre contrato, processo, partes, objeto ou data citadas na pergunta.
3. Só responda se encontrar o valor EXATO explicitamente indicado em UM dos contextos.
4. Não liste múltiplos valores. Traga apenas a resposta correta, exatamente como aparece no texto, junto ao número do contexto.
5. Se não encontrar o valor solicitado em nenhum contexto, responda: "A informação solicitada não foi encontrada no contexto fornecido."
6. NUNCA invente, deduza ou estime valores.

FORMATO OBRIGATÓRIO DA RESPOSTA:
- Se encontrou o valor:
  "
  Valor encontrado: R$ [valor exato]
  Trecho encontrado: "[trecho original]
  "

- Se não encontrou:
  A informação solicitada não foi encontrada no contexto fornecido."

**Contextos fornecidos:**
{context}

**Pergunta:**
{question}
"""

_cached_rag_chain = None

def build_rag_chain_fixed(llm, vector_store, pdf_directory, chunk_size, chunk_overlap, use_neighbor_retriever=True, k=1, neighbors=1, force_reload=False):
    """
    Constrói cadeia RAG usando componentes CORRIGIDOS
    """
    global _cached_rag_chain
    if _cached_rag_chain is not None and not force_reload:
        print("♻️ Usando RAG chain em cache")
        return _cached_rag_chain
    
    if not llm or not vector_store:
        print("❌ LLM ou Vector Store não fornecidos")
        return None

    try:
        if use_neighbor_retriever:
            print("🏠 Construindo RAG com Neighbor Retriever CORRIGIDO...")
            from .vector_store_manager import load_and_chunk_pdfs
            all_chunks = load_and_chunk_pdfs(pdf_directory, chunk_size, chunk_overlap)
            from .neighbor_retriever import SimpleNeighborRetriever
            retriever = SimpleNeighborRetriever(
                vector_store=vector_store,
                all_chunks=all_chunks,
                k=k,
                neighbors=neighbors
            )
            print(f"✅ Neighbor Retriever: k={k}, neighbors={neighbors}")
        else:
            print("🔍 Construindo RAG com retriever básico...")
            retriever = vector_store.as_retriever(search_kwargs={"k": k})

        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE_LEGAL,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
            verbose=False
        )
        
        print("✅ Cadeia RAG CORRIGIDA construída com sucesso")
        _cached_rag_chain = qa_chain
        return qa_chain
        
    except Exception as e:
        print(f"❌ Erro ao construir RAG: {e}")
        import traceback
        traceback.print_exc()
        return None

def build_rag_chain(llm, vector_store, pdf_directory, chunk_size, chunk_overlap, k=5, neighbors=1, force_reload=False):
    """Versão padrão com neighbor retriever"""
    return build_rag_chain_fixed(
        llm=llm,
        vector_store=vector_store,
        pdf_directory=pdf_directory,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_neighbor_retriever=True,
        k=k,
        neighbors=neighbors,
        force_reload=force_reload
    )

def build_rag_chain_basic(llm, vector_store, pdf_directory, chunk_size, chunk_overlap, k=5, force_reload=False):
    """Versão básica sem neighbor retriever"""
    return build_rag_chain_fixed(
        llm=llm,
        vector_store=vector_store,
        pdf_directory=pdf_directory,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_neighbor_retriever=False,
        k=k,
        neighbors=0,
        force_reload=force_reload
    )

def build_rag_chain_custom(llm, vector_store, pdf_directory, chunk_size, chunk_overlap, k=3, neighbors=1, force_reload=False):
    """Versão customizável para experimentos"""
    return build_rag_chain_fixed(
        llm=llm,
        vector_store=vector_store,
        pdf_directory=pdf_directory,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_neighbor_retriever=True,
        k=k,
        neighbors=neighbors,
        force_reload=force_reload
    )
