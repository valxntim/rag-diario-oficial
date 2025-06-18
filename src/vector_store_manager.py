# src/vector_store_manager.py
"""
Gerenciador para criar o Vector Store a partir de arquivos PDF.
ESTRATÉGIA SIMPLIFICADA: Usa o RecursiveCharacterTextSplitter padrão da indústria,
com parâmetros otimizados para os extratos de contrato.
"""
import os
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from .config import FAISS_INDEX_PATH, PDF_DIRECTORY, CHUNK_SIZE, CHUNK_OVERLAP
from .llm_interface import get_ollama_embeddings

_cached_vector_store = None

def load_and_chunk_pdfs(directory: str) -> list[Document]:
    """
    Carrega todos os PDFs de um diretório e os divide em chunks
    usando o RecursiveCharacterTextSplitter.
    """
    if not os.path.isdir(directory):
        print(f"ERRO: Diretório de PDFs não encontrado em '{directory}'")
        return []

    print(f"Carregando e processando PDFs do diretório: {directory}...")
    loader = PyPDFDirectoryLoader(directory, recursive=True)
    docs_from_pdfs = loader.load()

    print(f"Dividindo {len(docs_from_pdfs)} páginas em chunks (tamanho: {CHUNK_SIZE}, sobreposição: {CHUNK_OVERLAP})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(docs_from_pdfs)
    
    print(f"Total de {len(chunks)} chunks criados.")
    return chunks

def get_vector_store(force_recreate=False):
    """
    Carrega um índice FAISS existente ou cria um novo a partir dos PDFs.
    """
    global _cached_vector_store
    if _cached_vector_store is not None and not force_reload:
        return _cached_vector_store

    embeddings_model = get_ollama_embeddings()
    if not embeddings_model:
        return None

    if not force_recreate and os.path.exists(FAISS_INDEX_PATH):
        print(f"\nCarregando índice FAISS existente de: {FAISS_INDEX_PATH}")
        _cached_vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings_model,
            allow_dangerous_deserialization=True
        )
        print("Índice FAISS carregado com sucesso.")
        return _cached_vector_store
    
    print("\nCriando um novo índice FAISS a partir dos arquivos PDF (Estratégia: Splitter Recursivo)...")
    
    documents_to_index = load_and_chunk_pdfs(PDF_DIRECTORY)
    if not documents_to_index:
        print("Nenhum documento encontrado para indexar. Abortando.")
        return None

    print(f"Gerando embeddings para {len(documents_to_index)} chunks e construindo o índice FAISS...")
    start_time = time.time()
    vectorstore = FAISS.from_documents(
        documents=documents_to_index,
        embedding=embeddings_model
    )
    end_time = time.time()
    print(f"Novo índice FAISS criado com sucesso em {end_time - start_time:.2f} segundos.")
    
    print(f"Salvando o novo índice FAISS em: {FAISS_INDEX_PATH}")
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("Índice salvo com sucesso.")

    _cached_vector_store = vectorstore
    return vectorstore