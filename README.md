# 🤖 Sistema RAG para Análise do Diário Oficial
SLIDE[https://www.canva.com/design/DAGtMTcSSS0/QNUv00OOUQDFkRnVt4c4rw/edit?utm_content=DAGtMTcSSS0&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton]
Este repositório contém a implementação de um sistema de **Geração Aumentada por Recuperação (RAG)** projetado para extrair informações de documentos complexos e não estruturados, como o Diário Oficial do Distrito Federal (DODF).

O objetivo deste projeto é investigar e avaliar a eficácia de uma arquitetura RAG de linha de base ("baseline") para democratizar o acesso a informações públicas, transformando o denso conteúdo dos Diários em um formato de perguntas e respostas acessível ao cidadão comum.

---

## 🚀 Começando

Para rodar este projeto, você precisará ter o **Python 3.10+** e uma instância do **Ollama** (local ou remota) em execução.

### 1. Preparação do Ambiente

Siga estes passos no seu terminal para configurar o ambiente do projeto:

```bash
# 1. Clone o repositório
git clone [https://docs.github.com/articles/referencing-and-citing-content](https://docs.github.com/articles/referencing-and-citing-content)
cd [nome-da-pasta-do-repositorio]

# 2. Crie e ative um ambiente virtual
python3 -m venv venv
source venv/bin/activate

# 3. Instale todas as dependências necessárias
pip install -r requirements.txt
```

### 2. Configuração (O Passo Mais Importante!) ⚙️

Antes de executar, você **precisa** configurar seus modelos de IA. Toda a configuração está centralizada em um único arquivo: `src/config.py`.

Abra o arquivo `src/config.py` e ajuste as seguintes variáveis:

* `OLLAMA_LLM_URL` e `OLLAMA_LLM_MODEL`: Informe a URL do seu servidor Ollama e o nome do modelo de linguagem que será usado para gerar as respostas (ex: `llama4:latest`).
* `OLLAMA_EMBEDDING_URL` e `OLLAMA_EMBEDDING_MODEL`: Informe a URL e o nome do modelo de embedding que será usado para criar e consultar o índice vetorial (ex: `nomic-embed-text:latest`).
* `PDF_DIRECTORY`: O caminho para a pasta que contém os PDFs a serem analisados. Alguns PDFs de exemplo já estão incluídos em `data/pdfs/contratos` para um teste rápido.
* **Hiperparâmetros (Opcional):** Você também pode ajustar `CHUNK_SIZE`, `CHUNK_OVERLAP` e `RETRIEVER_SEARCH_K` neste arquivo para experimentar diferentes configurações do RAG.

---

## ▶️ Executando a Avaliação

O teste do sistema é simples e automatizado.

1.  **Garanta que sua instância do Ollama esteja rodando** e acessível a partir da máquina onde você executa o script.
2.  Execute o script de avaliação principal:

    ```bash
    python3 evaluate_rag.py
    ```

### O que o script faz?

* Ele irá primeiro carregar os PDFs da pasta configurada e, se um índice vetorial ainda não existir, irá criar um novo (isso pode levar um tempo na primeira execução).
* Em seguida, ele processará as **192 perguntas** do nosso dataset de avaliação (`dataset_verificado_final.jsonl`).
* Para cada pergunta, ele usará a cadeia RAG para buscar contextos e gerar uma resposta.

---

## 📊 Analisando os Resultados

Ao final da execução, o script imprimirá no terminal um resumo com a **acurácia final** do sistema.

Além disso, ele gerará um arquivo chamado `evaluation_results_baseline.csv` com uma análise detalhada de cada pergunta, contendo as seguintes colunas:

* `is_correct`: `True` ou `False`, indicando se a resposta foi considerada correta.
* `question`: A pergunta feita ao sistema.
* `ground_truth_answer`: A resposta correta (gabarito).
* `generated_answer`: A resposta exata que o sistema RAG forneceu.
* `retrieved_contexts`: **A coluna mais importante para análise!** Ela contém os trechos de texto exatos que o sistema recuperou para formular a resposta, permitindo diagnosticar falhas de busca.

Este arquivo CSV é a principal evidência gerada por este projeto, permitindo uma análise profunda das limitações e do potencial da arquitetura RAG base.
