# src/chatbot_cli.py
"""
Interface de Linha de Comando (CLI) para interagir com o ChatBot RAG.
"""
import time
from .config import EXAMPLE_QUESTIONS #, QUERY_TEST_RETRIEVAL # Teste de recuperação pode ser opcional aqui
from .llm_interface import get_ollama_llm
from .vector_store_manager import get_vector_store
from .rag_chain_builder import build_rag_chain

def run_chatbot():
    """
    Inicializa e executa o loop principal do chatbot.
    """
    print("--- Iniciando ChatBot RAG do Diário Oficial ---")

    # 1. Obter o Vector Store (carregar ou criar)
    # Para forçar a recriação do índice (ex: após adicionar novos PDFs):
    # vector_store = get_vector_store(force_recreate=True) 
    vector_store = get_vector_store() 
    if not vector_store:
        print("ERRO: Não foi possível inicializar o Vector Store. Encerrando.")
        return

    # --- Teste de Recuperação Simples (Opcional - pode ser removido ou comentado) ---
    # print("\n--- Teste de Recuperação Simples no VectorStore ---")
    # print(f"Realizando busca por: '{QUERY_TEST_RETRIEVAL}'")
    # try:
    #     relevant_docs_with_score = vector_store.similarity_search_with_score(QUERY_TEST_RETRIEVAL, k=2)
    #     if relevant_docs_with_score:
    #         print(f"\nOs {len(relevant_docs_with_score)} docs mais relevantes para '{QUERY_TEST_RETRIEVAL}':")
    #         for i, (doc, score) in enumerate(relevant_docs_with_score):
    #             print(f"  Doc {i+1} (Score: {score:.4f}) Origem: {doc.metadata.get('source', 'N/A')} Pág: {doc.metadata.get('page', -1)+1}")
    #     else:
    #         print("  Nenhum documento relevante encontrado para o teste de recuperação.")
    # except Exception as e_search:
    #     print(f"  ERRO durante a busca de similaridade para teste: {e_search}")
    # --- Fim do Teste de Recuperação ---

    # 2. Obter o LLM de Geração
    llm = get_ollama_llm()
    if not llm:
        print("ERRO: Não foi possível inicializar o LLM. Encerrando.")
        return

    # 3. Construir a Cadeia RAG
    qa_chain = build_rag_chain(llm, vector_store)
    if not qa_chain:
        print("ERRO: Não foi possível construir a cadeia RAG. Encerrando.")
        return

    print("\n--- ChatBot Pronto! ---")
    print("Digite 'sair' para terminar o chat.")

    if EXAMPLE_QUESTIONS:
        print("\nExperimente uma das seguintes perguntas (ou digite a sua):")
        for i, q_ex in enumerate(EXAMPLE_QUESTIONS):
            print(f"  {i+1}. {q_ex}")
    print("-" * 30)

    while True:
        user_question = input("\nSua pergunta: ")
        if user_question.lower() == 'sair':
            print("Encerrando o chat. Até logo!")
            break
        if not user_question.strip():
            # print("Por favor, digite uma pergunta.") # Opcional
            continue

        # print(f"Processando sua pergunta: '{user_question}'...") # Menos verboso
        start_time_rag = time.time()
        try:
            result = qa_chain.invoke({"query": user_question})
            end_time_rag = time.time()

            print(f"\nResposta ({end_time_rag - start_time_rag:.2f}s):") # Mais conciso
            print(result.get("result", "Nenhuma resposta foi gerada.").strip())

            # Opcional: Mostrar documentos fonte
            # print("\nDocumentos Fonte Recuperados:")
            # source_documents = result.get("source_documents", [])
            # if source_documents:
            #     for i, doc_source in enumerate(source_documents):
            #         print(f"  Fonte {i+1}: {doc_source.metadata.get('source', 'N/A')}, Pág: {doc_source.metadata.get('page', -1) + 1}")
            # else:
            #     print("  Nenhum documento fonte retornado.")

        except Exception as e_rag_query:
            print(f"ERRO ao processar a pergunta com a cadeia RAG: {e_rag_query}")

# O main.py chamará esta função
# if __name__ == "__main__":
# run_chatbot()
