# Arquivo: src/vector_store_manager.py (Versão Final Unificada com Ollama)

# Imports necessários
import os
import time
import fitz  # PyMuPDF
import torch
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from torch.nn.functional import cosine_similarity

# Import das configurações e da função de embedding centralizada
from src.config import PDF_DIRECTORY, FAISS_INDEX_PATH
from src.llm_interface import get_specialized_embeddings # <-- Usaremos a mesma função em todo lugar

# --- Configurações do Processamento de Documentos ---
# Agora são controladas centralmente pelo config.py e llm_interface.py

# Limiar para a quebra de tópicos.
SIMILARITY_THRESHOLD = 0.85

# Limite máximo de caracteres para qualquer chunk.
MAX_CHUNK_CHARACTERS = 512

# --- Variáveis de Cache ---
_cached_vector_store = None

def get_embedding_model_for_processing():
    """
    Esta função agora simplesmente chama a função centralizada do llm_interface
    para garantir que estamos usando o mesmo modelo de embedding em todo lugar.
    O modelo usado será o 'nomic-embed-text' do seu servidor Ollama.
    """
    print("Redirecionando para o modelo de embedding especializado via Ollama...")
    return get_specialized_embeddings()

def process_pdfs_with_HYBRID_segmentation() -> list[Document]:
    """
    Pipeline HÍBRIDA que usa o modelo de embedding do Ollama para criar os chunks.
    """
    embeddings_model = get_embedding_model_for_processing()
    all_final_chunks = []

    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith(".pdf")]
    print(f"Encontrados {len(pdf_files)} PDFs. Iniciando segmentação HÍBRIDA...")

    for pdf_filename in tqdm(pdf_files, desc="Processando PDFs"):
        pdf_path = os.path.join(PDF_DIRECTORY, pdf_filename)
        
        try:
            # ETAPA 1: Extrair parágrafos
            all_paragraphs = []
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                blocks = page.get_text("blocks", sort=True)
                for b in blocks:
                    text = b[4].strip().replace('\n', ' ')
                    if len(text) > 50:
                        all_paragraphs.append({"text": text, "page": page_num + 1})
            
            if not all_paragraphs:
                continue

            # ETAPA 2: Gerar embeddings com o modelo do Ollama
            paragraph_texts = [p["text"] for p in all_paragraphs]
            embeddings_list = embeddings_model.embed_documents(paragraph_texts)
            embeddings = torch.tensor(embeddings_list, dtype=torch.float32)

            # ETAPA 3: Calcular fronteiras semânticas
            boundaries = [0]
            for i in range(1, len(embeddings)):
                sim_tensor = cosine_similarity(embeddings[i - 1].unsqueeze(0), embeddings[i].unsqueeze(0))
                sim = sim_tensor.item()
                if sim < SIMILARITY_THRESHOLD:
                    boundaries.append(i)
            boundaries.append(len(all_paragraphs))

            # ETAPA 4: Agrupar parágrafos em chunks com limite de tamanho
            for i in range(len(boundaries) - 1):
                semantic_group_paragraphs = all_paragraphs[boundaries[i]:boundaries[i+1]]
                current_chunk_paragraphs = []
                current_chunk_len = 0
                for paragraph in semantic_group_paragraphs:
                    paragraph_len = len(paragraph["text"])
                    if current_chunk_len > 0 and (current_chunk_len + paragraph_len + 2) > MAX_CHUNK_CHARACTERS:
                        chunk_text = "\n\n".join(p["text"] for p in current_chunk_paragraphs)
                        pages = sorted(list(set(p["page"] for p in current_chunk_paragraphs)))
                        page_label = f"{pages[0]}" if len(pages) == 1 else f"{pages[0]}-{pages[-1]}"
                        metadata = {"source": pdf_filename, "pages": page_label}
                        all_final_chunks.append(Document(page_content=chunk_text, metadata=metadata))
                        current_chunk_paragraphs = [paragraph]
                        current_chunk_len = paragraph_len
                    else:
                        current_chunk_paragraphs.append(paragraph)
                        current_chunk_len += paragraph_len + 2
                if current_chunk_paragraphs:
                    chunk_text = "\n\n".join(p["text"] for p in current_chunk_paragraphs)
                    pages = sorted(list(set(p["page"] for p in current_chunk_paragraphs)))
                    page_label = f"{pages[0]}" if len(pages) == 1 else f"{pages[0]}-{pages[-1]}"
                    metadata = {"source": pdf_filename, "pages": page_label}
                    all_final_chunks.append(Document(page_content=chunk_text, metadata=metadata))
                    
        except Exception as e:
            print(f"Erro ao processar {pdf_filename}: {e}")
            continue

    print(f"Segmentação concluída. Total de {len(all_final_chunks)} chunks semânticos criados.")
    return all_final_chunks


def get_vector_store(force_recreate=False):
    """
    Carrega ou cria um índice FAISS.
    """
    global _cached_vector_store
    if _cached_vector_store is not None and not force_recreate:
        return _cached_vector_store

    embeddings_model_for_storage = get_embedding_model_for_processing()

    if not force_recreate and os.path.exists(FAISS_INDEX_PATH):
        print(f"\nCarregando índice FAISS existente de: {FAISS_INDEX_PATH}")
        try:
            # Precisamos do embedding_model aqui para carregar o índice
            _cached_vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings_model_for_storage, allow_dangerous_deserialization=True)
            print("Índice FAISS carregado com sucesso.")
            return _cached_vector_store
        except Exception as e:
            print(f"AVISO: Não foi possível carregar o índice local. Erro: {e}. Recriando do zero.")

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