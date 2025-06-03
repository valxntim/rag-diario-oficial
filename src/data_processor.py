# src/data_processor.py
"""
Funções para carregar e processar documentos PDF.
"""
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .config import CHUNK_SIZE, CHUNK_OVERLAP

def load_and_process_pdfs(pdf_dir_path: str) -> list:
    """
    Carrega todos os arquivos PDF de um diretório especificado,
    extrai o texto e divide o texto de cada PDF em chunks menores.
    Retorna uma lista de todos os chunks (objetos Document) de todos os PDFs.
    """
    all_chunks_from_all_pdfs = []
    
    if not os.path.isdir(pdf_dir_path):
        print(f"AVISO: O diretório de PDFs '{pdf_dir_path}' não foi encontrado.")
        return all_chunks_from_all_pdfs

    pdf_files = [f for f in os.listdir(pdf_dir_path) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"AVISO: Nenhum arquivo PDF encontrado no diretório '{pdf_dir_path}'.")
        return all_chunks_from_all_pdfs

    print(f"Encontrados {len(pdf_files)} arquivos PDF em '{pdf_dir_path}'. Iniciando processamento...")

    for filename in pdf_files:
        pdf_path = os.path.join(pdf_dir_path, filename)
        # print(f"Processando PDF: {pdf_path}...") # Print mais verboso, pode ser mantido se desejar
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            if not pages:
                # print(f"  Nenhuma página carregada de {filename}.") # Menos verboso
                continue

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                add_start_index=True,
            )
            
            chunks_from_this_pdf = text_splitter.split_documents(pages)
            all_chunks_from_all_pdfs.extend(chunks_from_this_pdf)
            # print(f"  PDF '{filename}' processado: {len(pages)} páginas, {len(chunks_from_this_pdf)} chunks gerados.") # Print mais verboso
            
        except Exception as e:
            print(f"  Erro ao carregar ou processar o PDF {pdf_path}: {e}")
            continue 
    
    print(f"Processamento de PDFs concluído. Total de chunks gerados: {len(all_chunks_from_all_pdfs)}")
    return all_chunks_from_all_pdfs

