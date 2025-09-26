# vector_store_manager.py
# VERSÃO FINAL CORRIGIDA - Baseada na sua versão original (mais rápida!)

import os
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from .config import FAISS_INDEX_PATH, PDF_DIRECTORY, CHUNK_SIZE, CHUNK_OVERLAP
from .llm_interface import get_embeddings

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
        add_start_index=True  # Para neighbor retriever
    )
    
    chunks = text_splitter.split_documents(docs_from_pdfs)
    
    # Adicionar chunk_id para neighbor retriever funcionar
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        chunk.metadata['source_doc'] = chunk.metadata.get('source', 'unknown')
    
    print(f"Total de {len(chunks)} chunks criados.")
    return chunks

def get_vector_store(force_recreate=False):
    """
    Carrega um índice FAISS existente ou cria um novo a partir dos PDFs.
    VERSÃO OTIMIZADA: Usa FAISS.from_documents() - muito mais rápido!
    """
    global _cached_vector_store
    if _cached_vector_store is not None and not force_recreate:  # ← CORRIGIDO: era force_reload
        return _cached_vector_store

    embeddings_model = get_embeddings()  # ← CORRIGIDO: era get_ollama_embeddings
    if not embeddings_model:
        return None

    if not force_recreate and os.path.exists(FAISS_INDEX_PATH):
        print(f"\n🔹 Carregando índice FAISS existente de: {FAISS_INDEX_PATH}")
        _cached_vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings_model,
            allow_dangerous_deserialization=True
        )
        print("Índice FAISS carregado com sucesso.")
        return _cached_vector_store
    
    print(f"\n🔹 Criando novo índice FAISS: {FAISS_INDEX_PATH}")
    print("🚀 Usando FAISS.from_documents() - método otimizado!")
    
    documents_to_index = load_and_chunk_pdfs(PDF_DIRECTORY)
    if not documents_to_index:
        print("Nenhum documento encontrado para indexar. Abortando.")
        return None

    print(f"⚡ Gerando embeddings para {len(documents_to_index)} chunks...")
    start_time = time.time()
    
    # FAISS.from_documents é MUITO mais rápido que manual loop!
    vectorstore = FAISS.from_documents(
        documents=documents_to_index,
        embedding=embeddings_model
    )
    
    end_time = time.time()
    print(f"✅ Novo índice FAISS criado em {end_time - start_time:.2f} segundos.")
    
    print(f"💾 Salvando índice em: {FAISS_INDEX_PATH}")
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("✅ Índice salvo com sucesso.")

    _cached_vector_store = vectorstore
    return vectorstore