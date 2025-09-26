from tqdm import tqdm
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List, Optional
import pickle
import os

# Compat Pydantic v1/v2 para retrievers customizados
try:
    from pydantic import ConfigDict, PrivateAttr
    HAS_V2 = True
except Exception:
    HAS_V2 = False
    from pydantic import PrivateAttr

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
_cached_bm25_retriever = None

# Retrievers finos com suporte Pydantic

if HAS_V2:
    class TopKLimiter(BaseRetriever):
        base: BaseRetriever
        k_final: int = 2
        model_config = ConfigDict(arbitrary_types_allowed=True)

        def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            docs = self.base.get_relevant_documents(query)
            return docs[: self.k_final]

        async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            docs = await self.base.aget_relevant_documents(query)
            return docs[: self.k_final]

    class NeighborExpander(BaseRetriever):
        base: BaseRetriever
        neighbors: int = 1
        _all_chunks: List[Document] = PrivateAttr(default_factory=list)
        model_config = ConfigDict(arbitrary_types_allowed=True)

        def __init__(self, base: BaseRetriever, all_chunks: List[Document], neighbors: int = 1, **data):
            super().__init__(base=base, neighbors=neighbors, **data)
            self._all_chunks = all_chunks

        @staticmethod
        def _get_chunk_index(doc: Document) -> Optional[int]:
            for key in ["chunk_id", "index", "chunk_idx", "i", "idx"]:
                if key in (doc.metadata or {}):
                    try:
                        return int(doc.metadata[key])
                    except Exception:
                        continue
            return None

        def _expand(self, seed_docs: List[Document]) -> List[Document]:
            out: List[Document] = []
            seen = set()

            def add_doc(d: Document):
                sig = (
                    (d.metadata or {}).get("source"),
                    (d.metadata or {}).get("chunk_id"),
                    hash(d.page_content[:128]) if d.page_content else None,
                )
                if sig not in seen:
                    seen.add(sig)
                    out.append(d)

            for d in seed_docs:
                add_doc(d)
                cid = self._get_chunk_index(d)
                if cid is None:
                    continue
                for off in range(1, self.neighbors + 1):
                    for nb in (cid - off, cid + off):
                        if 0 <= nb < len(self._all_chunks):
                            add_doc(self._all_chunks[nb])
            return out

        def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            base_docs = self.base.get_relevant_documents(query)
            return self._expand(base_docs)

        async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            base_docs = await self.base.aget_relevant_documents(query)
            return self._expand(base_docs)
else:
    class TopKLimiter(BaseRetriever):
        base: BaseRetriever
        k_final: int = 2

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            docs = self.base.get_relevant_documents(query)
            return docs[: self.k_final]

        async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            docs = await self.base.aget_relevant_documents(query)
            return docs[: self.k_final]

    class NeighborExpander(BaseRetriever):
        base: BaseRetriever
        neighbors: int = 1
        _all_chunks: List[Document] = PrivateAttr(default_factory=list)

        class Config:
            arbitrary_types_allowed = True

        def __init__(self, base: BaseRetriever, all_chunks: List[Document], neighbors: int = 1, **data):
            super().__init__(base=base, neighbors=neighbors, **data)
            self._all_chunks = all_chunks

        @staticmethod
        def _get_chunk_index(doc: Document) -> Optional[int]:
            for key in ["chunk_id", "index", "chunk_idx", "i", "idx"]:
                if key in (doc.metadata or {}):
                    try:
                        return int(doc.metadata[key])
                    except Exception:
                        continue
            return None

        def _expand(self, seed_docs: List[Document]) -> List[Document]:
            out: List[Document] = []
            seen = set()

            def add_doc(d: Document):
                sig = (
                    (d.metadata or {}).get("source"),
                    (d.metadata or {}).get("chunk_id"),
                    hash(d.page_content[:128]) if d.page_content else None,
                )
                if sig not in seen:
                    seen.add(sig)
                    out.append(d)

            for d in seed_docs:
                add_doc(d)
                cid = self._get_chunk_index(d)
                if cid is None:
                    continue
                for off in range(1, self.neighbors + 1):
                    for nb in (cid - off, cid + off):
                        if 0 <= nb < len(self._all_chunks):
                            add_doc(self._all_chunks[nb])
            return out

        def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            base_docs = self.base.get_relevant_documents(query)
            return self._expand(base_docs)

        async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            base_docs = await self.base.aget_relevant_documents(query)
            return self._expand(base_docs)

def get_or_create_bm25_retriever(all_chunks, force_reload=False, per_source_k: int = None):
    global _cached_bm25_retriever
    bm25_cache_file = os.path.join(PDF_DIRECTORY, "bm25_retriever.pkl")
    if _cached_bm25_retriever is not None and not force_reload:
        print("♻️ Usando BM25Retriever em cache (memória)")
        if per_source_k is not None:
            _cached_bm25_retriever.k = per_source_k
        return _cached_bm25_retriever
    if os.path.exists(bm25_cache_file) and not force_reload:
        try:
            print("📂 Carregando BM25Retriever do arquivo...")
            with open(bm25_cache_file, 'rb') as f:
                _cached_bm25_retriever = pickle.load(f)
            if per_source_k is not None:
                _cached_bm25_retriever.k = per_source_k
            print("✅ BM25Retriever carregado do arquivo")
            return _cached_bm25_retriever
        except Exception as e:
            print(f"⚠️ Erro ao carregar BM25 do arquivo: {e}")
    print("🔨 Criando novo BM25Retriever...")
    docs = all_chunks if hasattr(all_chunks[0], 'page_content') else [Document(page_content=str(c)) for c in all_chunks]
    bm25_retriever = BM25Retriever.from_documents(docs)
    if per_source_k is not None:
        bm25_retriever.k = per_source_k
    _cached_bm25_retriever = bm25_retriever
    try:
        with open(bm25_cache_file, 'wb') as f:
            pickle.dump(bm25_retriever, f)
        print("💾 BM25Retriever salvo em arquivo para próximas execuções")
    except Exception as e:
        print(f"⚠️ Erro ao salvar BM25: {e}")
    print("✅ BM25Retriever criado")
    return bm25_retriever

def build_rag_chain_fixed(
    llm,
    vector_store,
    use_neighbor_retriever=True,
    k=3,
    neighbors=1,
    force_reload=False,
    use_hybrid=False,
    hybrid_per_source_k: Optional[int] = None,
    final_top_k: Optional[int] = None
):
    global _cached_rag_chain
    if _cached_rag_chain is not None and not force_reload:
        print("♻️ Usando RAG chain em cache")
        return _cached_rag_chain

    if not llm or not vector_store:
        print("❌ LLM ou Vector Store não fornecidos")
        return None

    try:
        all_chunks = load_and_chunk_pdfs(PDF_DIRECTORY)
        per_source_k = hybrid_per_source_k if hybrid_per_source_k is not None else max(10, k)
        topN = final_top_k if final_top_k is not None else k

        if use_hybrid:
            print("🔀 Construindo RAG HÍBRIDO (FAISS + BM25) com fusão → top-N → vizinhos...")
            faiss_retriever_base = vector_store.as_retriever(search_kwargs={"k": per_source_k})
            bm25_retriever = get_or_create_bm25_retriever(all_chunks, force_reload, per_source_k=per_source_k)
            fused = EnsembleRetriever(retrievers=[faiss_retriever_base, bm25_retriever], weights=[0.6, 0.4])
            print(f"✅ Ensemble: FAISS k={per_source_k} + BM25 k={per_source_k} (pesos 0.6/0.4)")
            limited = TopKLimiter(base=fused, k_final=topN)
            print(f"✅ Limitador pós-fusão: top-N={topN}")
            if use_neighbor_retriever and neighbors > 0:
                retriever = NeighborExpander(base=limited, all_chunks=all_chunks, neighbors=neighbors)
                print(f"✅ Expansão de vizinhos: neighbors=±{neighbors}")
            else:
                retriever = limited
                print("✅ Sem expansão de vizinhos")
        else:
            from .neighbor_retriever import SimpleNeighborRetriever
            if use_neighbor_retriever:
                retriever = SimpleNeighborRetriever(
                    vector_store=vector_store, all_chunks=all_chunks, k=k, neighbors=neighbors
                )
            else:
                retriever = vector_store.as_retriever(search_kwargs={"k": k})

        # PROMPT: Corrigido para ChatPromptTemplate e kwargs explícitos
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Você é um assistente especializado em encontrar valores monetários em contratos públicos do Diário Oficial brasileiro."),
            ("human", PROMPT_TEMPLATE_LEGAL)
        ])
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt,
                "document_variable_name": "context",
            },
            verbose=False
        )

        print("✅ Cadeia RAG CORRIGIDA construída com sucesso")
        _cached_rag_chain = qa_chain
        return qa_chain

    except Exception as e:
        print(f"❌ Erro ao construir RAG: {e}")
        import traceback; traceback.print_exc()
        return None

# Assinaturas/funções de conveniência aqui...
# inclua as versões build_rag_chain_hybrid, build_rag_chain etc
