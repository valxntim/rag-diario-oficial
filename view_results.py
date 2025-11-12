"""
📊 DASHBOARD STREAMLIT - VISUALIZAÇÃO DE RESULTADOS RAG
======================================================
Interface interativa para explorar, filtrar e analisar resultados da avaliação do sistema RAG.

Funcionalidades:
- Carregamento de múltiplos arquivos JSONL de resultados
- Filtros por corretude (todos, corretos, incorretos)
- Busca por texto em pergunta/resposta/ID/PDF
- Paginação para navegar grandes datasets
- Visualização detalhada de cada resultado
- Estatísticas agregadas (acurácia, contextos, etc)
- Expanders para ver dados completos (extratos, contextos, JSON bruto)
"""

import streamlit as st
import json
from pathlib import Path
import glob
from typing import List, Dict, Any
import re
import datetime


# ========== CONFIGURAÇÃO DA PÁGINA STREAMLIT ==========
# Define layout, título e ícone da página
st.set_page_config(
    page_title="RAG Results (Simple)",  # Abas do navegador
    page_icon="📊",                     # Ícone na aba
    layout="wide"                       # Layout em 2+ colunas
)


# ========== ESTILOS CUSTOMIZADOS ==========
# CSS para melhorar a aparência dos elementos de texto (monospace, scrollável)
st.markdown("""
<style>
/* Caixa de código com fundo escuro e texto legível */
.box-mono {
  background-color: #111827;           /* Fundo escuro (cinza quase preto) */
  color: #e5e7eb;                     /* Texto claro */
  padding: 0.75rem;                   /* Espaçamento interno */
  border-radius: 0.25rem;             /* Cantos levemente arredondados */
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: 0.9em;                   /* Texto um pouco menor */
  line-height: 1.4;                   /* Espaçamento entre linhas */
  white-space: pre-wrap;              /* Preserva espaços em branco */
  border: 1px solid #374151;          /* Borda sutil */
}
/* Classe para fazer caixas de texto "scrolláveis" (max-height com overflow) */
.scroll { 
  max-height: 420px;                  /* Altura máxima */
  overflow-y: auto;                   /* Scrollbar vertical se ultrapassar */
}
</style>
""", unsafe_allow_html=True)


# ========== TÍTULO E HEADER ==========
st.title("📊 RAG Results (Simple)")
st.markdown("---")


# ========== FUNÇÕES AUXILIARES (HELPERS) ==========

def extract_numeric_difference(diff_value: Any) -> float:
    """
    Extrai valor numérico do campo 'diferenca_valor'.
    
    Estratégia:
    1. Se None: retorna 0.0
    2. Se já é número: converte para float
    3. Se é string: busca padrão de números, pega o primeiro
    4. Se não encontrar: retorna 0.0
    
    Argumentos:
        diff_value: Valor do campo diferenca_valor (pode ser qualquer coisa)
    
    Retorna:
        float: Valor numérico (ou 0.0 se não conseguir extrair)
    
    Exemplo:
        extract_numeric_difference("value_mismatch (expected=100.00, generated=50.00)")
        → 100.0 (primeiro número encontrado)
    """
    if diff_value is None:
        return 0.0
    
    # Se já é número, converte direto
    if isinstance(diff_value, (int, float)):
        return float(diff_value)
    
    # Se é string, tenta extrair número
    if isinstance(diff_value, str):
        # Procura por padrão de número (com ponto ou vírgula)
        nums = re.findall(r'[\d.,]+', diff_value)
        if nums:
            try:
                # Pega o primeiro número encontrado e remove vírgula
                return float(nums[0].replace(',', ''))
            except Exception:
                return 0.0
    
    return 0.0


def context_count(rec: Dict[str, Any]) -> int:
    """
    Conta quantos contextos foram recuperados para um resultado.
    
    Argumentos:
        rec: Dict com resultado (esperado ter campo 'contextos_recuperados')
    
    Retorna:
        int: Número de contextos (0 se não houver ou for vazio)
    """
    ctx = rec.get("contextos_recuperados", [])
    return len(ctx) if isinstance(ctx, list) else 0


# ========== CARREGAMENTO DE DADOS (COM CACHE) ==========
# @st.cache_data faz o Streamlit cachear (guardar em memória) o resultado
# Só recarrega se o arquivo mudar (detecta mudanças automaticamente)

@st.cache_data
def list_jsonl_files() -> List[str]:
    """
    Lista todos os arquivos JSONL de resultados disponíveis.
    
    Procura em: resultados/resultados_base_maior/*.jsonl
    
    Retorna:
        List[str]: Lista de caminhos de arquivos (ordenados)
    """
    return sorted(glob.glob("resultados/resultados_base_maior/*.jsonl"))


@st.cache_data
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Carrega arquivo JSONL (1 JSON por linha).
    
    Cada linha é um resultado (pergunta + resposta esperada + resposta gerada + métricas).
    Se tiver linha inválida, continua (não trava).
    
    Argumentos:
        path: Caminho do arquivo JSONL
    
    Retorna:
        List[Dict]: Lista de resultados (cada um é um dict)
    """
    recs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):  # i começa em 1
            s = line.strip()
            if not s:  # Pula linhas vazias
                continue
            try:
                rec = json.loads(s)  # Converte JSON string para dict
                rec["_line_number"] = i  # Adiciona número da linha para debug
                recs.append(rec)
            except json.JSONDecodeError as e:
                # Se linha é inválida, ainda registra para mostrar o erro
                recs.append({"_line_number": i, "_parse_error": str(e)})
    
    return recs


def compute_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calcula estatísticas agregadas dos resultados.
    
    Métricas:
    - total: Número de resultados
    - correct: Quantos acertaram
    - incorrect: Quantos erraram
    - accuracy: Porcentagem de acertos
    - avg_diff: Diferença média de valores
    - max_diff: Diferença máxima de valores
    
    Argumentos:
        records: Lista de resultados (dicts com 'acerto' e 'diferenca_valor')
    
    Retorna:
        Dict: Dicionário com as 6 métricas calculadas
    """
    total = len(records)
    correct = sum(1 for r in records if bool(r.get("acerto", False)))
    incorrect = total - correct
    accuracy = (correct / total * 100.0) if total else 0.0

    # Extrai diferenças numéricas de todos os resultados
    diffs = [extract_numeric_difference(r.get("diferenca_valor")) for r in records]
    avg_diff = (sum(diffs) / len(diffs)) if diffs else 0.0
    max_diff = max(diffs) if diffs else 0.0

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": accuracy,
        "avg_diff": avg_diff,
        "max_diff": max_diff,
    }


def render_record(r: Dict[str, Any], idx: int):
    """
    Renderiza (desenha) UM resultado na página Streamlit.
    
    Mostra:
    1. Header com ID, status, contextos, PDF
    2. Pergunta
    3. Lado a lado: resposta esperada vs resposta gerada
    4. 3 métricas: diferença numérica, sistema, timestamp
    5. 3 expanders: extrato, contextos, JSON bruto
    
    Argumentos:
        r: Dict com 1 resultado
        idx: Índice do resultado (para identificação)
    """
    # Extrai dados principais do resultado
    rec_id = r.get("id_versao_pergunta", f"rec-{idx}")
    is_correct = bool(r.get("acerto", False))
    pdf_name = str(r.get("pdf", "N/A"))
    sistema = str(r.get("sistema", "N/A"))
    cntx = context_count(r)

    # ========== ROW 1: HEADER COM INFORMAÇÕES RÁPIDAS ==========
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    with c1:
        st.markdown(f"### 📝 `{rec_id}`")  # ID em monospace
    with c2:
        st.write("✅ Correto" if is_correct else "❌ Incorreto")  # Status
    with c3:
        st.write(f"📄 {cntx} contextos")  # Número de contextos
    with c4:
        # PDF name truncado se muito longo (para não quebrar layout)
        short_pdf = pdf_name if len(pdf_name) <= 22 else pdf_name[:22] + "…"
        st.write(f"🗂️ {short_pdf}")

    # ========== ROW 2: PERGUNTA ==========
    st.markdown("**🤔 Pergunta:**")
    st.write(r.get("pergunta", "N/A"))

    # ========== ROW 3: RESPOSTAS LADO A LADO ==========
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🎯 Resposta Esperada:**")
        # Usa HTML bruto para estilo customizado (caixa escura)
        st.markdown(
            f'<div class="box-mono">{r.get("resposta_esperada","N/A")}</div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown("**🤖 Resposta Gerada:**")
        # Adiciona classe .scroll para fazer scrollável se muito longo
        st.markdown(
            f'<div class="box-mono scroll">{r.get("resposta_gerada","N/A")}</div>',
            unsafe_allow_html=True
        )

    # ========== ROW 4: TRÊS MÉTRICAS ==========
    c1, c2, c3 = st.columns(3)
    with c1:
        num = extract_numeric_difference(r.get("diferenca_valor"))
        st.info(f"💰 Diferença numérica: R$ {num:,.2f}")
    with c2:
        st.info(f"⚙️ Sistema: {sistema}")
    with c3:
        ts = r.get("timestamp", 0)
        if ts:
            # Converte timestamp Unix para hora legível
            dt = datetime.datetime.fromtimestamp(ts)
            st.info(f"⏰ {dt.strftime('%H:%M:%S')}")

    # ========== ROW 5: EXPANDERS (DETALHES OPCIONAIS) ==========
    c1, c2 = st.columns(2)
    with c1:
        # Expander = seção que pode ser expandida/colapsada pelo usuário
        with st.expander("📄 Extrato (completo)", expanded=False):
            extrato = r.get("extrato", "Não disponível")
            # Se extrato for dict, converte para JSON string legível
            if not isinstance(extrato, str):
                extrato = json.dumps(extrato, ensure_ascii=False, indent=2)
            st.markdown(
                f'<div class="box-mono scroll">{extrato}</div>',
                unsafe_allow_html=True
            )
    with c2:
        with st.expander("🔍 Contextos Recuperados (completo)", expanded=False):
            ctx = r.get("contextos_recuperados", [])
            if isinstance(ctx, list) and ctx:
                # Mostra cada contexto com número
                for i, c in enumerate(ctx, 1):
                    st.markdown(f"**Contexto {i}:**")
                    if not isinstance(c, str):
                        c = json.dumps(c, ensure_ascii=False, indent=2)
                    st.markdown(
                        f'<div class="box-mono scroll">{c}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.warning("⚠️ Nenhum contexto recuperado disponível")

    # Expander para ver JSON bruto (útil para debug)
    with st.expander("🧾 JSON bruto", expanded=False):
        st.code(json.dumps(r, ensure_ascii=False, indent=2), language="json")

    # Linha de separação
    st.markdown("---")


# ========== INTERFACE PRINCIPAL ==========

# Lista todos os arquivos JSONL disponíveis
files = list_jsonl_files()

if not files:
    # Se nenhum arquivo encontrado, mostra aviso
    st.warning("📂 Nenhum JSONL encontrado em ./results")
else:
    # ========== SELETOR DE ARQUIVO ==========
    selected = st.selectbox(
        "📁 Arquivo:",
        options=files,
        format_func=lambda p: Path(p).name  # Mostra só o nome, não o path completo
    )
    records = load_jsonl(selected)

    # ========== FILTROS E BUSCA ==========
    # Filtro por corretude (todos / corretos / incorretos)
    show_filter = st.selectbox(
        "Mostrar:",
        ["Todos", "Corretos", "Incorretos"],
        index=0
    )

    # Caixa de busca por texto
    q = st.text_input(
        "🔎 Buscar em pergunta/resposta/id/pdf/sistema:",
        ""
    )

    # Seletor de registros por página (para paginação)
    items_per_page = st.selectbox(
        "Registros por página:",
        [5, 10, 20, 50, 100],
        index=2  # Padrão = 20
    )

    # Seletor de página
    page = st.number_input(
        "Página:",
        min_value=1,
        step=1,
        value=1
    )

    # ========== FUNÇÃO DE FILTRO POR TEXTO ==========
    def matches(r: Dict[str, Any]) -> bool:
        """
        Verifica se um resultado contém o texto de busca em qualquer campo relevante.
        
        Campos buscados: pergunta, resposta_gerada, id_versao_pergunta, pdf, sistema
        """
        if not q:  # Se q vazio, tudo "combina"
            return True
        
        s = q.lower()  # Busca case-insensitive
        fields = [
            r.get("pergunta", ""),
            r.get("resposta_gerada", ""),
            r.get("id_versao_pergunta", ""),
            r.get("pdf", ""),
            r.get("sistema", ""),
        ]
        # Retorna True se o texto q aparece em algum dos campos
        return any(s in str(x).lower() for x in fields)

    # ========== APLICAR FILTROS ==========
    # Primeiro filtra por texto
    filtered = [r for r in records if matches(r)]

    # Depois filtra por corretude
    if show_filter == "Corretos":
        filtered = [r for r in filtered if bool(r.get("acerto", False))]
    elif show_filter == "Incorretos":
        filtered = [r for r in filtered if not bool(r.get("acerto", False))]

    # ========== ESTATÍSTICAS CONSOLIDADAS ==========
    stats = compute_stats(filtered)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📝 Total", stats["total"])
    c2.metric("✅ Corretos", stats["correct"])
    c3.metric("❌ Incorretos", stats["incorrect"])
    c4.metric("🎯 Precisão", f"{stats['accuracy']:.1f}%")
    c5.metric("💰 Dif. Média", f"R$ {stats['avg_diff']:.2f}")

    # ========== PAGINAÇÃO ==========
    # Calcula número total de páginas
    total_pages = max(1, (len(filtered) + items_per_page - 1) // items_per_page)
    
    # Se página selecionada > total de páginas, ajusta para última página
    page = min(page, total_pages)
    
    # Calcula índices para "fatiar" a lista
    start = (page - 1) * items_per_page
    end = min(start + items_per_page, len(filtered))
    
    # Mostra informação de paginação
    st.info(f"Mostrando {start+1}–{end} de {len(filtered)} | Página {page}/{total_pages}")

    # ========== RENDERIZAR RESULTADOS DA PÁGINA ATUAL ==========
    # Só mostra os resultados da página selecionada
    for i, rec in enumerate(filtered[start:end], start=start + 1):
        render_record(rec, i)