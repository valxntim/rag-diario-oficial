import os
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from .llm_interface import get_embeddings

_cached_vector_store = {}

def load_and_chunk_pdfs(directory: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """
    Carrega todos os PDFs de um diretório e os divide em chunks usando o chunk_size e chunk_overlap RECEBIDOS.
    """
    if not os.path.isdir(directory):
        print(f"ERRO: Diretório de PDFs não encontrado em '{directory}'")
        return []

    print(f"Carregando e processando PDFs do diretório: {directory}...")
    loader = PyPDFDirectoryLoader(directory, recursive=True)
    docs_from_pdfs = loader.load()

    print(f"Dividindo {len(docs_from_pdfs)} páginas em chunks (tamanho: {chunk_size}, sobreposição: {chunk_overlap})...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        add_start_index=True  # Para neighbor retriever
    )

    chunks = text_splitter.split_documents(docs_from_pdfs)

    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        chunk.metadata['source_doc'] = chunk.metadata.get('source', 'unknown')
    print(f"Total de {len(chunks)} chunks criados.")
    return chunks

def get_vector_store(faiss_index_path: str, pdf_directory: str, chunk_size: int, chunk_overlap: int, force_recreate: bool = False):
    """
    Carrega ou cria FAISS index, SEMPRE usando o path, diretório e chunk params passados como argumento!
    """
    cache_key = (faiss_index_path, chunk_size, chunk_overlap)
    global _cached_vector_store
    if cache_key in _cached_vector_store and not force_recreate:
        return _cached_vector_store[cache_key]

    embeddings_model = get_embeddings()
    if not embeddings_model:
        return None

    if not force_recreate and os.path.exists(faiss_index_path):
        print(f"\n🔹 Carregando índice FAISS existente de: {faiss_index_path}")
        vectorstore = FAISS.load_local(
            faiss_index_path,
            embeddings_model,
            allow_dangerous_deserialization=True
        )
        print("Índice FAISS carregado com sucesso.")
        _cached_vector_store[cache_key] = vectorstore
        return vectorstore

    print(f"\n🔹 Criando novo índice FAISS: {faiss_index_path}")
    print("🚀 Usando FAISS.from_documents()!")
    documents_to_index = load_and_chunk_pdfs(pdf_directory, chunk_size, chunk_overlap)
    if not documents_to_index:
        print("Nenhum documento encontrado para indexar. Abortando.")
        return None

    print(f"⚡ Gerando embeddings para {len(documents_to_index)} chunks...")
    start_time = time.time()
    vectorstore = FAISS.from_documents(
        documents=documents_to_index,
        embedding=embeddings_model
    )
    end_time = time.time()
    print(f"✅ Novo índice FAISS criado em {end_time - start_time:.2f} segundos.")
    print(f"💾 Salvando índice em: {faiss_index_path}")
    vectorstore.save_local(faiss_index_path)
    print("✅ Índice salvo com sucesso.")

    _cached_vector_store[cache_key] = vectorstore
    return vectorstore
