# src/vector_store_manager.py
"""
Gerenciador para criar, salvar e carregar o Vector Store (FAISS).
"""
import os
import time
from langchain_community.vectorstores import FAISS
from .config import FAISS_INDEX_PATH, PDF_DIRECTORY
from .data_processor import load_and_process_pdfs
from .llm_interface import get_ollama_embeddings

_cached_vector_store = None

def get_vector_store(force_recreate=False, force_reload_embeddings=False):
    """
    Carrega um índice FAISS existente ou cria um novo se não existir ou se force_recreate=True.
    Cacheia a instância do vector store.
    """
    global _cached_vector_store
    if _cached_vector_store is not None and not force_recreate:
        # print("Usando Vector Store FAISS do cache.") # Opcional
        return _cached_vector_store

    embeddings_model = get_ollama_embeddings(force_reload=force_reload_embeddings)
    if not embeddings_model:
        return None 

    vectorstore = None

    if not force_recreate and os.path.exists(FAISS_INDEX_PATH) and os.path.isdir(FAISS_INDEX_PATH):
        print(f"Tentando carregar índice FAISS existente de: {FAISS_INDEX_PATH}")
        try:
            start_time_load = time.time()
            vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH,
                embeddings_model,
                allow_dangerous_deserialization=True
            )
            end_time_load = time.time()
            print(f"Índice FAISS carregado com sucesso em {end_time_load - start_time_load:.2f} segundos.")
        except Exception as e:
            print(f"  Erro ao carregar índice FAISS: {e}. Tentando criar um novo.")
            vectorstore = None
    else:
        if force_recreate:
            print(f"Forçando recriação do índice FAISS.")
        else:
            print(f"Índice FAISS não encontrado em '{FAISS_INDEX_PATH}'. Será criado um novo.")

    if vectorstore is None:
        print("Iniciando a criação de um novo índice FAISS...")
        
        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True) # Garante que o diretório do índice exista
        # PDF_DIRECTORY é configurado em config.py, não precisa criar aqui se já existir
        # os.makedirs(PDF_DIRECTORY, exist_ok=True) 

        all_document_chunks = load_and_process_pdfs(PDF_DIRECTORY)
        
        if all_document_chunks:
            print(f"Gerando embeddings para {len(all_document_chunks)} chunks e construindo o índice FAISS...")
            start_time_create = time.time()
            try:
                vectorstore = FAISS.from_documents(
                    documents=all_document_chunks,
                    embedding=embeddings_model
                )
                end_time_create = time.time()
                print(f"Novo índice FAISS criado com sucesso em {end_time_create - start_time_create:.2f} segundos.")
                
                print(f"Salvando o novo índice FAISS em: {FAISS_INDEX_PATH}")
                try:
                    vectorstore.save_local(FAISS_INDEX_PATH)
                    print("  Índice FAISS salvo com sucesso.")
                except Exception as e_save:
                    print(f"  ERRO ao salvar o novo índice FAISS: {e_save}")
            except Exception as e_create_faiss:
                print(f"  ERRO ao criar o índice FAISS a partir dos documentos: {e_create_faiss}")
                return None
        else:
            print("Nenhum chunk foi gerado. O VectorStore não pôde ser criado.")
            return None

    _cached_vector_store = vectorstore
    return vectorstore

