# src/vector_store_manager.py (Versão Final com Chunking Hierárquico e Preciso)

import os
import time
import fitz  # PyMuPDF
import torch
import io
from PIL import Image
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from transformers import VisionEncoderDecoderModel, AutoProcessor
from huggingface_hub import snapshot_download

from src.config import FAISS_INDEX_PATH, PDF_DIRECTORY
from src.llm_interface import get_ollama_embeddings

_cached_vector_store = None
_cached_model_and_processor = None

def get_nougat_model_and_processor():
    """Carrega e armazena em cache o modelo e o processador Nougat para evitar recarregá-lo."""
    global _cached_model_and_processor
    if _cached_model_and_processor is not None:
        return _cached_model_and_processor

    print("Carregando o modelo e processador Nougat pela primeira vez...")
    MODEL_TAG = "facebook/nougat-base"
    
    local_model_path = snapshot_download(MODEL_TAG)
    
    model = VisionEncoderDecoderModel.from_pretrained(local_model_path)
    processor = AutoProcessor.from_pretrained(local_model_path)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    _cached_model_and_processor = (model, processor)
    print("Modelo e processador Nougat carregados.")
    return _cached_model_and_processor

def process_pdfs_with_nougat() -> list[Document]:
    """
    Processa todos os PDFs usando a pipeline hierárquica:
    1. PyMuPDF -> Nougat para extrair Markdown estruturado.
    2. MarkdownHeaderTextSplitter para dividir em seções lógicas.
    3. RecursiveCharacterTextSplitter para quebrar seções grandes em chunks menores e precisos.
    """
    model, processor = get_nougat_model_and_processor()
    all_chunks = []

    # ETAPA 1 DE CHUNKING (ESTRUTURA)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

    # ETAPA 2 DE CHUNKING (TAMANHO PRECISO) - AJUSTADO CONFORME SUA SUGESTÃO
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,      # <-- Menor para maior precisão e menos ruído
        chunk_overlap=100,     # <-- Overlap proporcional ao novo tamanho
        length_function=len
    )

    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith(".pdf")]
    print(f"Encontrados {len(pdf_files)} PDFs. Iniciando processamento com chunking hierárquico preciso...")

    for pdf_filename in tqdm(pdf_files, desc="Processando PDFs"):
        pdf_path = os.path.join(PDF_DIRECTORY, pdf_filename)
        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(tqdm(doc, desc=f"Páginas de {pdf_filename}", leave=False)):
            try:
                # Extração com Nougat
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_bytes))
                if image.mode != "RGB": image = image.convert("RGB")
                
                pixel_values = processor(images=image, padding=True, return_tensors="pt").pixel_values
                outputs = model.generate(
                    pixel_values.to(model.device),
                    min_length=1, max_length=model.config.max_length, use_cache=True,
                    do_sample=False, num_beams=1, bad_words_ids=[[processor.tokenizer.pad_token_id]],
                    return_dict_in_generate=True,
                )
                sequence = processor.batch_decode(outputs.sequences)[0]
                markdown_text = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "").replace(processor.tokenizer.bos_token, "")

                # Aplica o chunking hierárquico
                md_chunks = markdown_splitter.split_text(markdown_text)
                final_chunks = recursive_splitter.split_documents(md_chunks)
                
                for chunk in final_chunks:
                    chunk.metadata["source"] = pdf_filename
                    chunk.metadata["page"] = page_num + 1
                
                all_chunks.extend(final_chunks)
            except Exception as e:
                print(f"ERRO ao processar a página {page_num + 1} do arquivo {pdf_filename}: {e}")
                continue
    
    print(f"Processamento concluído. Total de {len(all_chunks)} chunks hierárquicos criados.")
    return all_chunks

def get_vector_store(force_recreate=False):
    """
    Carrega um índice FAISS existente ou cria um novo usando a pipeline Nougat.
    """
    global _cached_vector_store
    if _cached_vector_store is not None and not force_recreate:
        return _cached_vector_store

    embeddings_model = get_ollama_embeddings()
    if not embeddings_model: return None

    if not force_recreate and os.path.exists(FAISS_INDEX_PATH):
        print(f"\nCarregando índice FAISS existente de: {FAISS_INDEX_PATH}")
        _cached_vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings_model,
            allow_dangerous_deserialization=True
        )
        print("Índice FAISS carregado com sucesso.")
        return _cached_vector_store
    
    print("\nCriando um novo índice FAISS com a pipeline Nougat + Markdown + Recursive Splitter...")
    
    # CHAMA A NOVA FUNÇÃO DE PROCESSAMENTO
    documents_to_index = process_pdfs_with_nougat()
    
    if not documents_to_index:
        print("Nenhum documento encontrado para indexar. Abortando.")
        return None

    print(f"Gerando embeddings para {len(documents_to_index)} chunks e construindo o índice FAISS...")
    start_time = time.time()
    vectorstore = FAISS.from_documents(documents=documents_to_index, embedding=embeddings_model)
    end_time = time.time()
    print(f"Novo índice FAISS criado com sucesso em {end_time - start_time:.2f} segundos.")
    
    print(f"Salvando o novo índice FAISS em: {FAISS_INDEX_PATH}")
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("Índice salvo com sucesso.")

    _cached_vector_store = vectorstore
    return vectorstore