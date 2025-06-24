# Arquivo: src/vector_store_manager.py (Versão Final Refatorada)

import os
import time
from tqdm import tqdm
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from torch.nn.functional import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings # Import corrigido

# Configurações movidas para o topo para fácil ajuste
EMBEDDING_MODEL_NAME = "rufimelo/Legal-BERTimbau-sts-large" # RECOMENDAÇÃO: Use o modelo jurídico aqui!
SIMILARITY_THRESHOLD = 0.7  # Limiar para a quebra de tópicos, ajustável

_cached_vector_store = None
_cached_shared_embeddings = None

def get_embedding_model_for_processing():
    """Carrega e armazena em cache o modelo de embedding que será usado em todo o processo."""
    global _cached_shared_embeddings
    if _cached_shared_embeddings is None:
        print(f"Inicializando embedding model: {EMBEDDING_MODEL_NAME}")
        # Usamos o wrapper do LangChain para compatibilidade total
        _cached_shared_embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},  # Mude para 'cuda' se tiver GPU
            encode_kwargs={'normalize_embeddings': True} # Normalizar é bom para similaridade de cosseno
        )
    return _cached_shared_embeddings

def process_pdfs_with_HYBRID_segmentation() -> list[Document]:
    """
    Pipeline HÍBRIDA Refatorada:
    1. Extrai texto e layout com PyMuPDF, mantendo o número da página por parágrafo.
    2. Aplica a segmentação semântica para agrupar por tópico, preservando os metadados.
    """
    embeddings_model = get_embedding_model_for_processing()
    all_final_chunks = []

    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith(".pdf")]
    print(f"Encontrados {len(pdf_files)} PDFs. Iniciando segmentação HÍBRIDA refatorada...")

    for pdf_filename in tqdm(pdf_files, desc="Processando PDFs"):
        pdf_path = os.path.join(PDF_DIRECTORY, pdf_filename)
        
        try:
            # ETAPA 1: Coletar todos os parágrafos do documento com seus metadados
            all_paragraphs = []
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                blocks = page.get_text("blocks")
                blocks.sort(key=lambda b: (b[1], b[0]))
                for b in blocks:
                    text = b[4].strip()
                    if len(text) > 50: # Filtra blocos pequenos ou vazios
                        all_paragraphs.append({"text": text, "page": page_num + 1})
            
            if not all_paragraphs:
                continue

            # ETAPA 2: Gerar embeddings para a segmentação
            paragraph_texts = [p["text"] for p in all_paragraphs]
            # Usamos o método interno do wrapper LangChain, que retorna uma lista de listas
            embeddings = embeddings_model.client.encode(paragraph_texts, convert_to_tensor=True)

            # ETAPA 3: Calcular fronteiras semânticas
            boundaries = [0]
            for i in range(1, len(embeddings)):
                sim = cosine_similarity(embeddings[i - 1], embeddings[i], dim=0)
                if sim < SIMILARITY_THRESHOLD:
                    boundaries.append(i)
            boundaries.append(len(all_paragraphs))

            # ETAPA 4: Agrupar parágrafos em chunks, preservando metadados
            for i in range(len(boundaries) - 1):
                start_idx, end_idx = boundaries[i], boundaries[i + 1]
                
                # Pega a fatia de parágrafos para este chunk
                chunk_paragraphs = all_paragraphs[start_idx:end_idx]
                
                # Constrói o texto do chunk e os metadados
                chunk_text = "\n\n".join([p["text"] for p in chunk_paragraphs])
                
                pages = sorted(list(set([p["page"] for p in chunk_paragraphs])))
                page_label = f"{pages[0]}" if len(pages) == 1 else f"{pages[0]}-{pages[-1]}"
                
                metadata = {"source": pdf_filename, "pages": page_label}
                
                if len(chunk_text) > 100:
                    chunk = Document(page_content=chunk_text, metadata=metadata)
                    all_final_chunks.append(chunk)

        except Exception as e:
            print(f"Erro ao processar {pdf_filename}: {e}")
            continue

    print(f"Segmentação concluída. Total de {len(all_final_chunks)} chunks semânticos criados.")
    return all_final_chunks


def get_vector_store(force_recreate=False):
    """
    Carrega ou cria um índice FAISS, agora usando a mesma função de embedding em todo o processo.
    """
    global _cached_vector_store
    if _cached_vector_store is not None and not force_recreate:
        return _cached_vector_store

    # Usamos o mesmo modelo para tudo, garantindo consistência
    embeddings_model_for_storage = get_embedding_model_for_processing()

    if not embeddings_model_for_storage:
        return None

    if not force_recreate and os.path.exists(FAISS_INDEX_PATH):
        print(f"\nCarregando índice FAISS existente de: {FAISS_INDEX_PATH}")
        _cached_vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings_model_for_storage, allow_dangerous_deserialization=True)
        print("Índice FAISS carregado com sucesso.")
        return _cached_vector_store

    print("\nCriando um novo índice FAISS com segmentação semântica HÍBRIDA...")
    documents_to_index = process_pdfs_with_HYBRID_segmentation()
    
    if not documents_to_index:
        print("Nenhum documento encontrado para indexar. Abortando.")
        return None

    print(f"Gerando embeddings para {len(documents_to_index)} chunks e construindo o índice FAISS...")
    start_time = time.time()
    vectorstore = FAISS.from_documents(documents=documents_to_index, embedding=embeddings_model_for_storage)
    end_time = time.time()
    print(f"Novo índice FAISS criado com sucesso em {end_time - start_time:.2f} segundos.")
    
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("Índice salvo com sucesso.")
    _cached_vector_store = vectorstore
    return vectorstore