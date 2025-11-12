"""
CONSTRUTOR DE CADEIA RAG (Retrieval-Augmented Generation)
============================================================
Monta a pipeline completa que conecta:
  Embedding (query) → Busca no FAISS → Retriever → Prompt → LLM → Resposta

Oferece 3 funções com diferentes configurações:
- build_rag_chain: Com neighbor retriever (padrão recomendado)
- build_rag_chain_basic: Sem neighbors (mais rápido, menos contexto)
- build_rag_chain_custom: Totalmente customizável
"""

from tqdm import tqdm
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# ========== TEMPLATE DO PROMPT ==========
# Este é o "brain" que diz ao LLM como processar a pergunta e contexto
PROMPT_TEMPLATE_LEGAL = """
Você é um assistente especializado em encontrar valores monetários em contratos públicos do Diário Oficial brasileiro.

INSTRUÇÕES:
1. Analise cuidadosamente TODOS os contextos fornecidos.
2. Sua tarefa é identificar o valor monetário EXATO que responde à pergunta abaixo.
3. Considere: contrato, processo, partes, objeto ou data mencionadas na pergunta.
4. Só responda se encontrar o valor EXATO em UM dos contextos fornecidos.
5. Não liste múltiplos valores. Traga APENAS a resposta correta, exatamente como aparece no texto.
6. Se não encontrar o valor em nenhum contexto, responda: "A informação solicitada não foi encontrada no contexto fornecido."
7. NUNCA invente, deduza ou estime valores.

FORMATO OBRIGATÓRIO DA RESPOSTA:

Se encontrou o valor:
  "Valor encontrado: R$ [valor exato]
   Trecho encontrado: \"[trecho original do documento]\""

Se NÃO encontrou:
  "A informação solicitada não foi encontrada no contexto fornecido."

--------

CONTEXTOS FORNECIDOS:
{context}

PERGUNTA:
{question}
"""


# ========== CACHE GLOBAL ==========
# Armazena a chain já construída para não reconstruir a cada chamada
_cached_rag_chain = None


def build_rag_chain_fixed(
    llm, 
    vector_store, 
    pdf_directory, 
    chunk_size, 
    chunk_overlap, 
    use_neighbor_retriever=True, 
    k=1, 
    neighbors=1, 
    force_reload=False
):
    """
    Função CORE que constrói a cadeia RAG com flexibilidade máxima.
    
    Fluxo:
    1. Opcionalmente carrega chunks (se usando neighbor retriever)
    2. Cria o retriever (neighbor ou básico)
    3. Cria o prompt template
    4. Conecta retriever + LLM numa chain RetrievalQA
    5. Retorna a chain pronta para usar
    
    Argumentos:
        llm: Modelo LLM já inicializado (ex: OllamaLLM)
        vector_store: Índice FAISS já carregado
        pdf_directory: Caminho para PDFs (apenas se use neighbor retriever)
        chunk_size: Tamanho dos chunks
        chunk_overlap: Sobreposição dos chunks
        use_neighbor_retriever: Se True, usa neighbor retriever; senão usa básico
        k: Número de chunks a recuperar
        neighbors: Número de vizinhos de cada lado (só se use_neighbor_retriever=True)
        force_reload: Se True, reconstrói mesmo que esteja em cache
    
    Retorna:
        RetrievalQA chain pronta para .invoke({"query": "pergunta"})
    
    Levanta:
        Exception: Se algo falhar na construção
    """
    global _cached_rag_chain
    
    # Se já tem em cache e não quer recarregar, retorna o cache
    if _cached_rag_chain is not None and not force_reload:
        print("♻️ Usando RAG chain em cache")
        return _cached_rag_chain
    
    # Validação básica
    if not llm or not vector_store:
        print("❌ LLM ou Vector Store não fornecidos")
        return None

    try:
        # ========== ETAPA 1: CRIAR RETRIEVER ==========
        if use_neighbor_retriever:
            # Neighbor retriever: puxa chunks vizinhos para expandir contexto
            print("🏠 Construindo RAG com Neighbor Retriever CORRIGIDO...")
            from .vector_store_manager import load_and_chunk_pdfs
            from .neighbor_retriever import SimpleNeighborRetriever
            
            # Carrega todos os chunks em ordem (necessário para o neighbor retriever)
            all_chunks = load_and_chunk_pdfs(pdf_directory, chunk_size, chunk_overlap)
            
            # Cria o retriever que expande com vizinhos
            retriever = SimpleNeighborRetriever(
                vector_store=vector_store,
                all_chunks=all_chunks,
                k=k,
                neighbors=neighbors
            )
            print(f"✅ Neighbor Retriever: k={k}, neighbors={neighbors}")
        else:
            # Basic retriever: apenas busca vetorial, sem vizinhos
            print("🔍 Construindo RAG com retriever básico...")
            retriever = vector_store.as_retriever(search_kwargs={"k": k})

        # ========== ETAPA 2: CRIAR PROMPT ==========
        # Template que instrui o LLM como responder
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE_LEGAL,
            input_variables=["context", "question"]  # Variáveis a serem preenchidas
        )
        
        # ========== ETAPA 3: MONTAR A CHAIN ==========
        # RetrievalQA é a classe que encadeia: query → retriever → prompt → llm → resposta
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,                               # Modelo que vai gerar a resposta
            chain_type="stuff",                     # "stuff" = coloca tudo num só prompt
            retriever=retriever,                    # Recupera documentos relevantes
            return_source_documents=True,           # Retorna os documentos usados
            chain_type_kwargs={"prompt": prompt},   # Usa nosso template customizado
            verbose=False                           # Não printa logs detalhados
        )
        
        print("✅ Cadeia RAG CORRIGIDA construída com sucesso")
        _cached_rag_chain = qa_chain
        return qa_chain
        
    except Exception as e:
        # Se algo der errado, printa erro e stack trace para debug
        print(f"❌ Erro ao construir RAG: {e}")
        import traceback
        traceback.print_exc()
        return None


def build_rag_chain(llm, vector_store, pdf_directory, chunk_size, chunk_overlap, k=5, neighbors=1, force_reload=False):
    """
    Versão RECOMENDADA com neighbor retriever (melhor qualidade).
    
    Use esta função na maioria dos casos.
    """
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
    """
    Versão SEM neighbor retriever (mais rápida, menos contexto).
    
    Use se precisar de speed e não se importa de perder contexto.
    """
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
    """
    Versão CUSTOMIZÁVEL para experimentos.
    
    Use para testar diferentes configurações de k e neighbors.
    """
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