"""
GERENCIADOR DO VECTOR STORE (FAISS)
======================================
Responsável por:
- Carregar PDFs de um diretório
- Dividir em chunks com tamanho configurável
- Gerar embeddings usando o backend configurado
- Criar/carregar índice FAISS
- Cachear para evitar recriações

O FAISS é uma biblioteca de busca vetorial de alta performance do Facebook.
Permite encontrar os chunks mais similares à query em tempo O(1) ~
"""

import os
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from .llm_interface import get_embeddings


# ========== CACHE GLOBAL ==========
# Evita recarregar o mesmo vector store múltiplas vezes
# Chave: (caminho_do_índice, chunk_size, chunk_overlap)
_cached_vector_store = {}


def load_and_chunk_pdfs(directory: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """
    Carrega todos os PDFs de um diretório e os divide em chunks menores.
    
    Processo:
    1. Lê todos os PDFs recursivamente do diretório
    2. Usa RecursiveCharacterTextSplitter para dividir por tamanho
    3. Atribui chunk_id a cada chunk (índice na lista ordenada)
    
    Argumentos:
        directory: Caminho para pasta com PDFs
        chunk_size: Tamanho máximo de cada chunk (caracteres)
        chunk_overlap: Sobreposição entre chunks (caracteres)
    
    Retorna:
        Lista de Documents, cada um representando um chunk de texto
    
    Levanta:
        Exception se o diretório não existir
    """
    
    # Valida se a pasta existe
    if not os.path.isdir(directory):
        print(f"ERRO: Diretório de PDFs não encontrado em '{directory}'")
        return []

    # ========== ETAPA 1: CARREGAR PDFs ==========
    # Usa PyPDFDirectoryLoader que busca recursivamente por *.pdf
    print(f"Carregando e processando PDFs do diretório: {directory}...")
    loader = PyPDFDirectoryLoader(directory, recursive=True)
    docs_from_pdfs = loader.load()
    
    # Printa estatística de quantas páginas foram carregadas
    print(f"Dividindo {len(docs_from_pdfs)} páginas em chunks (tamanho: {chunk_size}, sobreposição: {chunk_overlap})...")

    # ========== ETAPA 2: DIVIDIR EM CHUNKS ==========
    # RecursiveCharacterTextSplitter tenta dividir de forma inteligente
    # Não quebra no meio de palavras; respeita estrutura
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,          # Tamanho máximo de cada chunk
        chunk_overlap=chunk_overlap,    # Sobreposição para manter contexto
        length_function=len,            # Como contar tamanho (caracteres)
        is_separator_regex=False,       # Separadores são strings literais, não regex
        add_start_index=True            # Adiciona índice de inicio do chunk no texto original
    )
    
    chunks = text_splitter.split_documents(docs_from_pdfs)

    # ========== ETAPA 3: ADICIONAR METADADOS ==========
    # Cada chunk recebe um identificador único (chunk_id)
    # Isso é crucial para o neighbor retriever funcionar
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i  # Índice na lista ordenada
        chunk.metadata['source_doc'] = chunk.metadata.get('source', 'unknown')
    
    print(f"Total de {len(chunks)} chunks criados.")
    return chunks


def get_vector_store(
    faiss_index_path: str, 
    pdf_directory: str, 
    chunk_size: int, 
    chunk_overlap: int, 
    force_recreate: bool = False
):
    """
    Carrega ou cria um índice FAISS para busca vetorial.
    
    Lógica:
    1. Se existe índice em cache E não quer forçar recriação → retorna cache
    2. Se existe índice no disco E não quer forçar → carrega do disco
    3. Senão: cria novo índice a partir dos PDFs
    
    Argumentos:
        faiss_index_path: Caminho completo do índice FAISS (ex: data/vector_store/faiss_index_chunk600)
        pdf_directory: Caminho dos PDFs a indexar
        chunk_size: Tamanho dos chunks
        chunk_overlap: Sobreposição dos chunks
        force_recreate: Se True, ignora cache e cria novo do zero
    
    Retorna:
        Objeto FAISS pronto para busca
    
    Levanta:
        Exception se houver erro ao criar embeddings ou carregar índice
    """
    
    # ========== ETAPA 1: VERIFICAR CACHE ==========
    cache_key = (faiss_index_path, chunk_size, chunk_overlap)
    global _cached_vector_store
    
    if cache_key in _cached_vector_store and not force_recreate:
        # Já tem em memória, retorna direto
        return _cached_vector_store[cache_key]

    # ========== ETAPA 2: INICIALIZAR MODELO DE EMBEDDING ==========
    # Obtém o backend de embedding (Ollama ou HuggingFace)
    embeddings_model = get_embeddings()
    if not embeddings_model:
        return None

    # ========== ETAPA 3: CARREGAR DO DISCO OU CRIAR NOVO ==========
    if not force_recreate and os.path.exists(faiss_index_path):
        # Índice já existe no disco, carrega
        print(f"\n🔹 Carregando índice FAISS existente de: {faiss_index_path}")
        vectorstore = FAISS.load_local(
            faiss_index_path,
            embeddings_model,
            allow_dangerous_deserialization=True  # Necessário para evitar aviso de segurança
        )
        print("Índice FAISS carregado com sucesso.")
        _cached_vector_store[cache_key] = vectorstore
        return vectorstore

    # ========== ETAPA 4: CRIAR NOVO ÍNDICE ==========
    print(f"\n🔹 Criando novo índice FAISS: {faiss_index_path}")
    print("🚀 Usando FAISS.from_documents()!")
    
    # Carrega e divide os PDFs em chunks
    documents_to_index = load_and_chunk_pdfs(pdf_directory, chunk_size, chunk_overlap)
    
    if not documents_to_index:
        print("Nenhum documento encontrado para indexar. Abortando.")
        return None

    # ========== ETAPA 5: GERAR EMBEDDINGS E INDEXAR ==========
    # Este é o passo mais LENTO: converte cada chunk em embedding (vetor)
    print(f"⚡ Gerando embeddings para {len(documents_to_index)} chunks...")
    start_time = time.time()
    
    vectorstore = FAISS.from_documents(
        documents=documents_to_index,
        embedding=embeddings_model
    )
    
    end_time = time.time()
    print(f"✅ Novo índice FAISS criado em {end_time - start_time:.2f} segundos.")
    
    # ========== ETAPA 6: SALVAR NO DISCO ==========
    # Salva o índice para reusar em próximas execuções (evita reindexar)
    print(f"💾 Salvando índice em: {faiss_index_path}")
    vectorstore.save_local(faiss_index_path)
    print("✅ Índice salvo com sucesso.")

    # Armazena em cache para evitar recarregar nesta sessão
    _cached_vector_store[cache_key] = vectorstore
    return vectorstore