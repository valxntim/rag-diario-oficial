# rag_chain_builder_fixed.py
# RAG Chain Builder usando componentes CORRIGIDOS
from tqdm import tqdm  # ← Make sure this import exists!

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from .config import RETRIEVER_SEARCH_K
from .vector_store_manager import load_and_chunk_pdfs, PDF_DIRECTORY

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
  "Pergunta: [pergunta original]
  Valor encontrado: R$ [valor exato]
  Contexto correspondente: [número do contexto]
  Trecho encontrado: "[trecho original]""

- Se não encontrou:
  "Pergunta: [pergunta original]
  A informação solicitada não foi encontrada no contexto fornecido."

**Contextos fornecidos:**
{context}

**Pergunta:**
{question}
"""


_cached_rag_chain = None

def build_rag_chain_fixed(llm, vector_store, use_neighbor_retriever=False, k=1, neighbors=0, force_reload=False):
    """
    Constrói cadeia RAG usando componentes CORRIGIDOS
    
    Args:
        llm: Modelo de linguagem
        vector_store: Vector store corrigido
        use_neighbor_retriever: Se True, usa retriever com vizinhos
        k: Número de chunks principais
        neighbors: Número de vizinhos antes/depois
        force_reload: Se True, recarrega a chain
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
            
            # Carregar chunks - IMPORTANTE: mesmos chunks do índice
            all_chunks = load_and_chunk_pdfs(PDF_DIRECTORY)
            
            # Usar neighbor retriever corrigido
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

        # Prompt otimizado para documentos legais
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE_LEGAL,
            input_variables=["context", "question"]
        )
        
        # Criar chain
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

# Funções de conveniência
def build_rag_chain(llm, vector_store, force_reload=False):
    """Versão padrão com neighbor retriever (recomendada)"""
    return build_rag_chain_fixed(
        llm=llm,
        vector_store=vector_store,
        use_neighbor_retriever=False,
        k=RETRIEVER_SEARCH_K,
        neighbors=0,
        force_reload=force_reload
    )

def build_rag_chain_with_neighbors(llm, vector_store, force_reload=False):
    """Versão com neighbor retriever (mesmo que build_rag_chain)"""
    return build_rag_chain(llm, vector_store, force_reload)

def build_rag_chain_basic(llm, vector_store, force_reload=False):
    """Versão básica sem neighbor retriever (para comparação)"""
    return build_rag_chain_fixed(
        llm=llm,
        vector_store=vector_store,
        use_neighbor_retriever=False,
        k=RETRIEVER_SEARCH_K,
        neighbors=0,
        force_reload=force_reload
    )

def build_rag_chain_custom(llm, vector_store, k=3, neighbors=1, force_reload=False):
    """Versão customizável para experimentos"""
    return build_rag_chain_fixed(
        llm=llm,
        vector_store=vector_store,
        use_neighbor_retriever=True,
        k=k,
        neighbors=neighbors,
        force_reload=force_reload
    )