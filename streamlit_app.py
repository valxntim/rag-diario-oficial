# streamlit_app.py
# 🚀 BEAUTIFUL RAG CHATBOT - Legal Contract Q&A System
# Built for TCC - Gustavo Valentim

import streamlit as st
import sys
import os
import time
from datetime import datetime
import json

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import your existing modules
try:
    from src.vector_store_manager import get_vector_store
    from src.rag_chain_builder import build_rag_chain_with_neighbors, build_rag_chain_basic
    from src.llm_interface import get_llm, get_embeddings
    from src.config import FAISS_INDEX_PATH, PDF_DIRECTORY
except ImportError as e:
    st.error(f"❌ Erro ao importar módulos: {e}")
    st.stop()

# 🎨 PAGE CONFIGURATION
st.set_page_config(
    page_title="🏛️ Legal RAG Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 CUSTOM CSS - Beautiful Dark Theme
st.markdown("""
<style>
    /* Main theme */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Chat message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        border-left: 4px solid #4facfe;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        border-left: 4px solid #00f2fe;
    }
    
    .context-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    .stats-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        color: white;
        margin: 0.25rem;
    }
    
    .example-question {
        background: rgba(71, 172, 254, 0.1);
        border: 1px solid #4facfe;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.3rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .example-question:hover {
        background: rgba(71, 172, 254, 0.2);
        transform: translateY(-2px);
    }
    
    /* Sidebar styling */
    .sidebar-content {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 🎯 INITIALIZE SESSION STATE
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'system_loaded' not in st.session_state:
    st.session_state.system_loaded = False

# 🏛️ MAIN HEADER
st.markdown('<h1 class="main-header">🏛️ Assistente RAG Legal</h1>', unsafe_allow_html=True)
st.markdown("### *Sistema Inteligente para Análise de Contratos do Diário Oficial*")

# 📊 SIDEBAR - CONTROLS & SETTINGS
with st.sidebar:
    st.markdown("## ⚙️ Configurações do Sistema")
    
    # System status
    if st.session_state.system_loaded:
        st.success("✅ Sistema Carregado")
    else:
        st.warning("⏳ Sistema não inicializado")
    
    st.markdown("---")
    
    # RAG Configuration
    st.markdown("### 🔧 Configurações RAG")
    
    use_neighbors = st.toggle(
        "🏠 Usar Neighbor Retriever", 
        value=True,
        help="Expande contexto com chunks vizinhos"
    )
    
    k_value = st.slider(
        "📊 Número de chunks (k)",
        min_value=1, max_value=10, value=3,
        help="Quantos chunks principais buscar"
    )
    
    if use_neighbors:
        neighbors_value = st.slider(
            "🔗 Vizinhos por chunk",
            min_value=0, max_value=3, value=1,
            help="Quantos vizinhos antes/depois"
        )
    else:
        neighbors_value = 0
    
    st.markdown("---")
    
    # System reload button
    if st.button("🔄 Recarregar Sistema", type="secondary"):
        st.session_state.system_loaded = False
        st.session_state.rag_system = None
        st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Limpar Conversa", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📈 Estatísticas")
    if st.session_state.system_loaded:
        if os.path.exists(FAISS_INDEX_PATH):
            st.metric("📁 Índice FAISS", "✅ Carregado")
        st.metric("💬 Mensagens", len(st.session_state.chat_history))

# 🚀 SYSTEM INITIALIZATION
if not st.session_state.system_loaded:
    st.markdown("## 🔧 Inicializando Sistema RAG...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Load Vector Store
        status_text.text("📁 Carregando Vector Store...")
        progress_bar.progress(25)
        vector_store = get_vector_store(force_recreate=False)
        
        if not vector_store:
            st.error("❌ Falha ao carregar Vector Store!")
            st.stop()
        
        # Step 2: Initialize LLM
        status_text.text("🧠 Inicializando LLM...")
        progress_bar.progress(50)
        llm = get_llm()
        
        if not llm:
            st.error("❌ Falha ao inicializar LLM!")
            st.stop()
        
        # Step 3: Build RAG Chain
        status_text.text("⚡ Construindo RAG Chain...")
        progress_bar.progress(75)
        
        if use_neighbors:
            rag_chain = build_rag_chain_with_neighbors(llm, vector_store)
        else:
            rag_chain = build_rag_chain_basic(llm, vector_store)
        
        if not rag_chain:
            st.error("❌ Falha ao construir RAG Chain!")
            st.stop()
        
        # Step 4: Complete
        status_text.text("✅ Sistema pronto!")
        progress_bar.progress(100)
        
        # Store in session state
        st.session_state.rag_system = rag_chain
        st.session_state.system_loaded = True
        
        time.sleep(1)  # Brief pause to show completion
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Erro durante inicialização: {e}")
        st.stop()

# 🎯 EXAMPLE QUESTIONS
if st.session_state.system_loaded and len(st.session_state.chat_history) == 0:
    st.markdown("## 💡 Perguntas de Exemplo")
    
    example_questions = [
        "Qual é o valor total do contrato nº 169/2019?",
        "Quem é o contratado no processo 305.000.016/2016?",
        "Qual a vigência do contrato assinado em outubro de 2019?",
        "Quantos contratos foram assinados com o BRB?",
        "Quais são os valores destinados à contratação de serviços?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(f"💬 {question}", key=f"example_{i}", use_container_width=True):
                # Add to chat and process
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": question,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                st.rerun()

# 💬 CHAT INTERFACE
if st.session_state.system_loaded:
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("## 💬 Conversa")
        
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'''
                <div class="user-message">
                    <strong>👤 Você ({message["timestamp"]}):</strong><br>
                    {message["content"]}
                </div>
                ''', unsafe_allow_html=True)
                
            else:  # assistant
                st.markdown(f'''
                <div class="bot-message">
                    <strong>🤖 Assistente ({message["timestamp"]}):</strong><br>
                    {message["content"]}
                </div>
                ''', unsafe_allow_html=True)
                
                # Show contexts if available
                if "contexts" in message:
                    with st.expander(f"📄 Contextos Recuperados ({len(message['contexts'])} chunks)", expanded=False):
                        for i, context in enumerate(message["contexts"], 1):
                            st.markdown(f'''
                            <div class="context-box">
                                <strong>📄 Contexto {i}:</strong><br>
                                <small>{context[:500]}{'...' if len(context) > 500 else ''}</small>
                            </div>
                            ''', unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    
    # Process pending question (from example questions)
    if (st.session_state.chat_history and 
        st.session_state.chat_history[-1]["role"] == "user" and 
        len(st.session_state.chat_history) % 2 == 1):
        
        user_question = st.session_state.chat_history[-1]["content"]
        
        with st.spinner("🤔 Pensando... Isso pode levar alguns segundos..."):
            try:
                # Run RAG query
                result = st.session_state.rag_system.invoke({"query": user_question})
                
                answer = result.get("result", "Não foi possível gerar uma resposta.")
                source_docs = result.get("source_documents", [])
                
                # Extract contexts
                contexts = [doc.page_content for doc in source_docs]
                
                # Add assistant response to chat
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "contexts": contexts,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                st.rerun()
                
            except Exception as e:
                error_msg = f"❌ Erro ao processar pergunta: {e}"
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                st.rerun()
    
    # New question input
    new_question = st.text_input(
        "✍️ Digite sua pergunta sobre contratos:",
        placeholder="Ex: Qual é o valor do contrato nº 123/2020?",
        key="new_question"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Enviar Pergunta", type="primary", use_container_width=True):
            if new_question.strip():
                # Add user message
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": new_question.strip(),
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                st.rerun()
            else:
                st.warning("⚠️ Digite uma pergunta primeiro!")

# 📊 FOOTER
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('''
    <div class="stats-metric">
        <strong>🎓 TCC Project</strong><br>
        <small>Legal RAG System</small>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown('''
    <div class="stats-metric">
        <strong>⚡ Powered by</strong><br>
        <small>Ollama + LangChain</small>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown('''
    <div class="stats-metric">
        <strong>🏛️ Domain</strong><br>
        <small>Brazilian Legal Docs</small>
    </div>
    ''', unsafe_allow_html=True)

st.markdown("---")
st.markdown("*🚀 Built with ❤️ by Gustavo Valentim - Universidade de Brasília (UnB)*")