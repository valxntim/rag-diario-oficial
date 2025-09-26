from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr

class NeighborRetriever:
    """Reliable neighbor retriever (guaranteed position-based neighbors!)"""
    def __init__(self, vector_store: FAISS, all_chunks: List[Document], k=3, neighbors=1):
        self.vector_store = vector_store
        self.all_chunks = all_chunks
        self.k = k
        self.neighbors = neighbors

    def expand_chunk_with_neighbors(self, chunk: Document) -> str:
        """ALWAYS get neighbors by position in array!"""
        # Trust chunk_id == array index (guaranteed if you set at chunk creation)
        idx = int(chunk.metadata['chunk_id'])
        start = max(0, idx - self.neighbors)
        end = min(len(self.all_chunks), idx + self.neighbors + 1)
        contents = []
        for i in range(start, end):
            contents.append(self.all_chunks[i].page_content)
        return "\n".join(contents)

    def get_relevant_documents(self, query: str) -> List[Document]:
        # Top-k retrieval as before
        results = self.vector_store.similarity_search_with_score(query, k=self.k)
        expanded_documents = []
        seen = set()
        for doc, score in results:
            idx = int(doc.metadata['chunk_id'])
            # Make unique by idx to avoid duplicate neighborhoods
            if idx in seen:
                continue
            seen.add(idx)
            expanded_content = self.expand_chunk_with_neighbors(doc)
            expanded_doc = Document(
                page_content=expanded_content,
                metadata={
                    **doc.metadata,
                    'expanded': True,
                    'original_score': score,
                    'neighbors_count': min(self.neighbors * 2 + 1, len(self.all_chunks))
                }
            )
            expanded_documents.append(expanded_doc)
        return expanded_documents

# Compatibility with LangChain RetrievalQA (Pydantic, no dynamic attributes!)
class SimpleNeighborRetriever(BaseRetriever):
    _neighbor_retriever: NeighborRetriever = PrivateAttr()

    def __init__(self, vector_store, all_chunks, k=3, neighbors=1):
        super().__init__()
        self._neighbor_retriever = NeighborRetriever(vector_store, all_chunks, k, neighbors)

    def _get_relevant_documents(self, query: str):
        return self._neighbor_retriever.get_relevant_documents(query)
