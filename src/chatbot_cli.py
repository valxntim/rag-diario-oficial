# src/chatbot_cli.py
"""
Interface de Linha de Comando (CLI) para interagir com o ChatBot RAG.
Este arquivo permanece praticamente o mesmo, pois a complexidade foi abstraída
para o rag_chain_builder.
"""
import time
# Imports relativos para a estrutura do projeto
from .config import EXAMPLE_QUESTIONS
from .llm_interface import get_ollama_llm
from .vector_store_manager import get_vector_store
from .rag_chain_builder import build_rag_chain

def run_chatbot():
    """
    Inicializa e executa o loop principal do chatbot.
    """
    print("--- Iniciando ChatBot RAG do Diário Oficial ---")

    # 1. Obter o Vector Store (carregar ou criar)
    #vector_store = get_vector_store() 
    vector_store = get_vector_store(force_recreate=True)
    if not vector_store:
        print("ERRO: Não foi possível inicializar o Vector Store. Encerrando.")
        return

    # 2. Obter o LLM de Geração
    llm = get_ollama_llm()
    if not llm:
        print("ERRO: Não foi possível inicializar o LLM. Encerrando.")
        return

    # 3. Construir a Cadeia RAG (agora usará o MultiQueryRetriever por baixo dos panos)
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
            continue

        print(f"Processando sua pergunta com MultiQueryRetriever: '{user_question}'...")
        start_time_rag = time.time()
        try:
            result = qa_chain.invoke({"query": user_question})
            end_time_rag = time.time()

            print(f"\nResposta ({end_time_rag - start_time_rag:.2f}s):")
            print(result.get("result", "Nenhuma resposta foi gerada.").strip())

            # Opcional: Mostrar documentos fonte para depuração
            print("\nDocumentos Fonte Recuperados:")
            source_documents = result.get("source_documents", [])
            if source_documents:
                # O MultiQueryRetriever pode retornar muitos documentos, vamos mostrar os metadados dos primeiros
                unique_sources = {}
                for doc_source in source_documents:
                    source_key = (
                        doc_source.metadata.get('source', 'N/A'),
                        doc_source.metadata.get('page', -1)
                    )
                    if source_key not in unique_sources:
                        unique_sources[source_key] = doc_source
                
                print(f"  Fontes únicas recuperadas: {len(unique_sources)}")
                for i, doc_source in enumerate(list(unique_sources.values())[:5]): # Mostra até 5 fontes únicas
                    print(f"  Fonte {i+1}: {doc_source.metadata.get('source', 'N/A')}, Pág: {doc_source.metadata.get('page', -1) + 1}")
                    # Descomente para ver o conteúdo do chunk
                    # print(f"      Conteúdo: \"{doc_source.page_content[:200]}...\"")
            else:
                print("  Nenhum documento fonte retornado.")

        except Exception as e_rag_query:
            print(f"ERRO ao processar a pergunta com a cadeia RAG: {e_rag_query}")

# O main.py chamará esta função
if __name__ == "__main__":
    run_chatbot()
