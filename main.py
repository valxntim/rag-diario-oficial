# main.py (na raiz do projeto, ao lado da pasta 'src' e 'data')
"""
Ponto de entrada principal para executar o ChatBot RAG.
"""
import sys
import os

# Adiciona o diretório raiz do projeto (o diretório que contém 'main.py' e a pasta 'src')
# ao início do sys.path. Isso permite que 'src' seja importado como um pacote.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.chatbot_cli import run_chatbot # Agora deve encontrar o pacote 'src'

if __name__ == "__main__":
    print("Iniciando a aplicação ChatBot RAG do Diário Oficial...")
    run_chatbot()
    print("Aplicação ChatBot RAG finalizada.")

