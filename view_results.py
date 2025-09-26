
import streamlit as st
import json
import pandas as pd
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="RAG Evaluation Results Viewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}

.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 0.75rem;
    border-radius: 0.25rem;
    margin: 0.5rem 0;
}

.error-box {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
    padding: 0.75rem;
    border-radius: 0.25rem;
    margin: 0.5rem 0;
}

.context-box {
    background-color: #2c3e50;
    border-left: 4px solid #3498db;
    color: #ecf0f1;
    padding: 0.75rem;
    margin: 0.5rem 0;
    border-radius: 0.25rem;
    font-family: monospace;
    font-size: 0.9em;
    line-height: 1.4;
}

.extrato-box {
    background-color: #34495e;
    border: 1px solid #7f8c8d;
    color: #ecf0f1;
    padding: 0.75rem;
    margin: 0.5rem 0;
    border-radius: 0.25rem;
    font-family: monospace;
    font-size: 0.9em;
    line-height: 1.4;
    white-space: pre-wrap;
}

.context-box:hover, .extrato-box:hover {
    background-color: #1a252f;
    transition: background-color 0.3s ease;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 RAG Evaluation Results Viewer")
st.markdown("---")

@st.cache_data
def load_jsonl(path):
    """Load JSONL file and return list of records"""
    try:
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    record['_line_number'] = line_num
                    records.append(record)
                except json.JSONDecodeError as e:
                    st.warning(f"Error parsing line {line_num}: {e}")
                    continue
        return records
    except FileNotFoundError:
        st.error(f"File not found: {path}")
        return []
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return []

def calculate_statistics(records):
    """Calculate evaluation statistics"""
    if not records:
        return {}

    total = len(records)
    correct = sum(1 for r in records if r.get('acerto', False))
    incorrect = total - correct
    accuracy = (correct / total) * 100 if total > 0 else 0

    # Value differences statistics
    value_diffs = [r.get('diferenca_valor', 0) for r in records if r.get('diferenca_valor') is not None]
    avg_diff = sum(value_diffs) / len(value_diffs) if value_diffs else 0

    # PDF distribution
    pdf_counts = {}
    for r in records:
        pdf = r.get('pdf', 'Unknown')
        pdf_counts[pdf] = pdf_counts.get(pdf, 0) + 1

    return {
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'accuracy': accuracy,
        'avg_diff': avg_diff,
        'pdf_counts': pdf_counts,
        'value_diffs': value_diffs
    }

def display_record(record, index):
    """Display a single evaluation record"""
    record_id = record.get('id_versao_pergunta', f'Record {index}')
    is_correct = record.get('acerto', False)

    # Main record container
    with st.container():
        # Header with result indicator
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.markdown(f"### 📝 {record_id}")

        with col2:
            if is_correct:
                st.markdown('<div class="success-box">✅ CORRETO</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-box">❌ INCORRETO</div>', unsafe_allow_html=True)

        with col3:
            pdf_name = record.get('pdf', 'N/A')[:20] + '...' if len(record.get('pdf', '')) > 20 else record.get('pdf', 'N/A')
            st.info(f"📄 {pdf_name}")

        # Question and answers
        st.markdown("**🤔 Pergunta:**")
        st.markdown(f"> {record.get('pergunta', 'N/A')}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🎯 Resposta Esperada:**")
            expected = record.get('resposta_esperada', 'N/A')
            st.markdown(f'<div class="success-box">{expected}</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("**🤖 Resposta Gerada:**")
            generated = record.get('resposta_gerada', 'N/A')
            box_class = "success-box" if is_correct else "error-box"
            st.markdown(f'<div class="{box_class}">{generated}</div>', unsafe_allow_html=True)

        # Value difference if available
        if record.get('diferenca_valor') is not None:
            diff_val = record.get('diferenca_valor', 0)
            st.metric("💰 Diferença de Valor", f"R$ {diff_val:,.2f}")

        # Expandable sections for detailed content
        with st.expander("📄 Ver Extrato do Documento", expanded=False):
            extrato = record.get('extrato', 'Não disponível')
            st.markdown(f'<div class="extrato-box">{extrato}</div>', unsafe_allow_html=True)

        with st.expander("🔍 Ver Contextos Recuperados", expanded=False):
            contextos = record.get('contextos_recuperados', [])
            if contextos:
                for i, contexto in enumerate(contextos):
                    st.markdown(f"**Contexto {i+1}:**")
                    st.markdown(f'<div class="context-box">{contexto}</div>', unsafe_allow_html=True)
            else:
                st.info("Nenhum contexto recuperado disponível")

        st.markdown("---")

# Sidebar for file loading and controls
with st.sidebar:
    st.header("🔧 Controles")

    # File input
    datafile = st.text_input(
        "📁 Arquivo JSONL", 
        value="./evaluation_results_hibrida_1300_200_1.jsonl",
        help="Caminho para o arquivo JSONL com os resultados"
    )

    load_button = st.button("🔄 Carregar Arquivo", type="primary")

    if load_button and datafile:
        # Clear cache and load new file
        load_jsonl.clear()
        st.session_state['records'] = load_jsonl(datafile)
        st.session_state['file_loaded'] = True
        st.success("Arquivo carregado!")

# Initialize session state
if 'records' not in st.session_state:
    st.session_state['records'] = []
if 'file_loaded' not in st.session_state:
    st.session_state['file_loaded'] = False

# Load default file on startup
if not st.session_state['file_loaded'] and Path(datafile if 'datafile' in locals() else "./evaluation_results_final.jsonl").exists():
    st.session_state['records'] = load_jsonl(datafile if 'datafile' in locals() else "./evaluation_results_final.jsonl")
    st.session_state['file_loaded'] = True

records = st.session_state.get('records', [])

if records:
    # Calculate and display statistics
    stats = calculate_statistics(records)

    st.header("📊 Estatísticas Gerais")

    # Main metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📝 Total de Registros", stats['total'])

    with col2:
        st.metric("✅ Respostas Corretas", stats['correct'])

    with col3:
        st.metric("❌ Respostas Incorretas", stats['incorrect'])

    with col4:
        st.metric("🎯 Precisão", f"{stats['accuracy']:.1f}%")

    # Additional statistics
    if stats['value_diffs']:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💰 Diferença Média de Valor", f"R$ {stats['avg_diff']:.2f}")
        with col2:
            st.metric("📄 Arquivos PDF Únicos", len(stats['pdf_counts']))

    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Filtros")

        # Filter by correctness
        show_filter = st.selectbox(
            "Mostrar:",
            ["Todos", "Apenas Corretos", "Apenas Incorretos"],
            index=0
        )

        # Filter by PDF
        pdf_files = ["Todos"] + list(stats['pdf_counts'].keys())
        selected_pdf = st.selectbox("Filtrar por PDF:", pdf_files, index=0)

        # Search in questions
        search_query = st.text_input("🔎 Buscar na pergunta:", "")

        # Value difference filter
        if stats['value_diffs']:
            show_value_diff = st.checkbox("Apenas com diferença de valor")
        else:
            show_value_diff = False

    # Apply filters
    filtered_records = records.copy()

    if show_filter == "Apenas Corretos":
        filtered_records = [r for r in filtered_records if r.get('acerto', False)]
    elif show_filter == "Apenas Incorretos":
        filtered_records = [r for r in filtered_records if not r.get('acerto', False)]

    if selected_pdf != "Todos":
        filtered_records = [r for r in filtered_records if r.get('pdf') == selected_pdf]

    if search_query:
        filtered_records = [r for r in filtered_records 
                          if search_query.lower() in r.get('pergunta', '').lower()]

    if show_value_diff:
        filtered_records = [r for r in filtered_records 
                          if r.get('diferenca_valor') is not None]

    # Display filtered results
    st.header(f"📋 Resultados ({len(filtered_records)} de {len(records)})")

    if filtered_records:
        # Pagination
        items_per_page = st.selectbox("Registros por página:", [5, 10, 20, 50], index=1)
        total_pages = (len(filtered_records) + items_per_page - 1) // items_per_page

        if total_pages > 1:
            page = st.selectbox("Página:", range(1, total_pages + 1), index=0)
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(filtered_records))
            page_records = filtered_records[start_idx:end_idx]

            st.info(f"Mostrando registros {start_idx + 1}-{end_idx} de {len(filtered_records)}")
        else:
            page_records = filtered_records

        # Display records
        for i, record in enumerate(page_records):
            display_record(record, start_idx + i + 1 if 'start_idx' in locals() else i + 1)

        # Export options
        with st.sidebar:
            st.header("📤 Exportar")
            if st.button("💾 Baixar Dados Filtrados (CSV)"):
                # Create a simplified dataframe for export
                export_data = []
                for record in filtered_records:
                    export_data.append({
                        'ID': record.get('id_versao_pergunta', ''),
                        'Pergunta': record.get('pergunta', ''),
                        'Resposta_Esperada': record.get('resposta_esperada', ''),
                        'Resposta_Gerada': record.get('resposta_gerada', ''),
                        'Acerto': record.get('acerto', False),
                        'PDF': record.get('pdf', ''),
                        'Diferenca_Valor': record.get('diferenca_valor', 0)
                    })

                df = pd.DataFrame(export_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="evaluation_results_filtered.csv",
                    mime="text/csv"
                )
    else:
        st.warning("Nenhum registro encontrado com os filtros aplicados.")

else:
    st.info("👆 Carregue um arquivo JSONL usando a barra lateral para começar.")

    # Show example of expected format
    with st.expander("📖 Formato de arquivo esperado"):
        st.markdown("""
        O arquivo JSONL deve conter uma linha por registro, com a seguinte estrutura:

        ```json
        {
            "id_versao_pergunta": "id_da_pergunta",
            "pergunta": "Qual é a pergunta?",
            "resposta_esperada": "Resposta esperada",
            "resposta_gerada": "Resposta gerada pelo modelo",
            "acerto": true,
            "pdf": "nome_do_arquivo.pdf",
            "extrato": "Texto extraído do documento...",
            "contextos_recuperados": ["contexto1", "contexto2"],
            "diferenca_valor": 0.0
        }
        ```
        """)
