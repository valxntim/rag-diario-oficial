# src/chatbot_cli.py
"""
Interface de Linha de Comando (CLI) para interagir com o ChatBot RAG.
Permite testes interativos e qualitativos do sistema.
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
    print("--- Iniciando ChatBot RAG do Diário Oficial (Versão de Teste Interativo) ---")

    # 1. Carrega o Vector Store já existente
    # Garante que não recriamos o índice a cada vez que o chat inicia
    vector_store = get_vector_store() 
    if not vector_store:
        print("ERRO: Não foi possível inicializar o Vector Store. Encerrando.")
        return

    # 2. Carrega o LLM de Geração
    llm = get_ollama_llm()
    if not llm:
        print("ERRO: Não foi possível inicializar o LLM. Encerrando.")
        return

    # 3. Constrói a cadeia RAG (usando a versão atual do builder, seja ela simples ou avançada)
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

        print(f"Processando sua pergunta: '{user_question}'...")
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
                for i, doc in enumerate(source_documents):
                    source = doc.metadata.get('source', 'N/A')
                    page = doc.metadata.get('page', -1)
                    print(f"  Fonte {i+1}: {os.path.basename(source)}, Pág: {page}")
                    # print(f"    Conteúdo: \"{doc.page_content[:200]}...\"") # Descomente para ver o conteúdo
            else:
                print("  Nenhum documento fonte retornado.")

        except Exception as e_rag_query:
            print(f"ERRO ao processar a pergunta com a cadeia RAG: {e_rag_query}")

# Para rodar este arquivo diretamente: python3 -m src.chatbot_cli
if __name__ == "__main__":
    run_chatbot()