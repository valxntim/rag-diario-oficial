import streamlit as st
import json
from pathlib import Path
import glob
from typing import List, Dict, Any
import re
import datetime

st.set_page_config(page_title="RAG Results (Simple)", page_icon="📊", layout="wide")

# Minimal styling for readable scroll areas
st.markdown("""
<style>
.box-mono {
  background-color: #111827; color: #e5e7eb;
  padding: 0.75rem; border-radius: 0.25rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: 0.9em; line-height: 1.4; white-space: pre-wrap; border: 1px solid #374151;
}
.scroll { max-height: 420px; overflow-y: auto; }
</style>
""", unsafe_allow_html=True)

st.title("📊 RAG Results (Simple)")
st.markdown("---")

# -------- Helpers --------
def extract_numeric_difference(diff_value: Any) -> float:
    """
    Parse numeric difference from diferenca_valor.
    No 'exact match' special-casing; non-numeric strings yield 0.0.
    """
    if diff_value is None:
        return 0.0
    if isinstance(diff_value, (int, float)):
        return float(diff_value)
    if isinstance(diff_value, str):
        nums = re.findall(r'[\d.,]+', diff_value)
        if nums:
            try:
                return float(nums[0].replace(',', ''))
            except Exception:
                return 0.0
    return 0.0

def context_count(rec: Dict[str, Any]) -> int:
    ctx = rec.get("contextos_recuperados", [])
    return len(ctx) if isinstance(ctx, list) else 0

# -------- Cached IO --------
@st.cache_data
def list_jsonl_files() -> List[str]:
    # Only from ./results folder
    return sorted(glob.glob("results/*.jsonl"))

@st.cache_data
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
                rec["_line_number"] = i
                recs.append(rec)
            except json.JSONDecodeError as e:
                # Keep going; mark parse error as a record for debugging
                recs.append({"_line_number": i, "_parse_error": str(e)})
    return recs

def compute_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    correct = sum(1 for r in records if bool(r.get("acerto", False)))
    incorrect = total - correct
    accuracy = (correct / total * 100.0) if total else 0.0

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
    rec_id = r.get("id_versao_pergunta", f"rec-{idx}")
    is_correct = bool(r.get("acerto", False))
    pdf_name = str(r.get("pdf", "N/A"))
    sistema = str(r.get("sistema", "N/A"))
    cntx = context_count(r)

    c1, c2, c3, c4 = st.columns([3,1,1,1])
    with c1:
        st.markdown(f"### 📝 `{rec_id}`")
    with c2:
        st.write("✅ Correto" if is_correct else "❌ Incorreto")
    with c3:
        st.write(f"📄 {cntx} contextos")
    with c4:
        short_pdf = pdf_name if len(pdf_name) <= 22 else pdf_name[:22] + "…"
        st.write(f"🗂️ {short_pdf}")

    st.markdown("**🤔 Pergunta:**")
    st.write(r.get("pergunta", "N/A"))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🎯 Resposta Esperada:**")
        st.markdown(f'<div class="box-mono">{r.get("resposta_esperada","N/A")}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown("**🤖 Resposta Gerada:**")
        st.markdown(f'<div class="box-mono scroll">{r.get("resposta_gerada","N/A")}</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        num = extract_numeric_difference(r.get("diferenca_valor"))
        st.info(f"💰 Diferença numérica: R$ {num:,.2f}")
    with c2:
        st.info(f"⚙️ Sistema: {sistema}")
    with c3:
        ts = r.get("timestamp", 0)
        if ts:
            dt = datetime.datetime.fromtimestamp(ts)
            st.info(f"⏰ {dt.strftime('%H:%M:%S')}")

    c1, c2 = st.columns(2)
    with c1:
        with st.expander("📄 Extrato (completo)", expanded=False):
            extrato = r.get("extrato", "Não disponível")
            if not isinstance(extrato, str):
                extrato = json.dumps(extrato, ensure_ascii=False, indent=2)
            st.markdown(f'<div class="box-mono scroll">{extrato}</div>', unsafe_allow_html=True)
    with c2:
        with st.expander("🔍 Contextos Recuperados (completo)", expanded=False):
            ctx = r.get("contextos_recuperados", [])
            if isinstance(ctx, list) and ctx:
                for i, c in enumerate(ctx, 1):
                    st.markdown(f"**Contexto {i}:**")
                    if not isinstance(c, str):
                        c = json.dumps(c, ensure_ascii=False, indent=2)
                    st.markdown(f'<div class="box-mono scroll">{c}</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Nenhum contexto recuperado disponível")

    with st.expander("🧾 JSON bruto", expanded=False):
        st.code(json.dumps(r, ensure_ascii=False, indent=2), language="json")

    st.markdown("---")

# -------- UI --------
files = list_jsonl_files()
if not files:
    st.warning("📂 Nenhum JSONL encontrado em ./results")
else:
    selected = st.selectbox("📁 Arquivo:", options=files, format_func=lambda p: Path(p).name)
    records = load_jsonl(selected)

    # New correctness filter
    show_filter = st.selectbox("Mostrar:", ["Todos", "Corretos", "Incorretos"], index=0)

    # Basic text search
    q = st.text_input("🔎 Buscar em pergunta/resposta/id/pdf/sistema:", "")
    items_per_page = st.selectbox("Registros por página:", [5, 10, 20, 50, 100], index=2)
    page = st.number_input("Página:", min_value=1, step=1, value=1)

    # Text filter
    def matches(r: Dict[str, Any]) -> bool:
        if not q:
            return True
        s = q.lower()
        fields = [
            r.get("pergunta",""),
            r.get("resposta_gerada",""),
            r.get("id_versao_pergunta",""),
            r.get("pdf",""),
            r.get("sistema",""),
        ]
        return any(s in str(x).lower() for x in fields)

    filtered = [r for r in records if matches(r)]

    # Apply correctness filter
    if show_filter == "Corretos":
        filtered = [r for r in filtered if bool(r.get("acerto", False))]
    elif show_filter == "Incorretos":
        filtered = [r for r in filtered if not bool(r.get("acerto", False))]

    # Simple stats
    stats = compute_stats(filtered)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📝 Total", stats["total"])
    c2.metric("✅ Corretos", stats["correct"])
    c3.metric("❌ Incorretos", stats["incorrect"])
    c4.metric("🎯 Precisão", f"{stats['accuracy']:.1f}%")
    c5.metric("💰 Dif. Média", f"R$ {stats['avg_diff']:.2f}")

    # Pagination
    total_pages = max(1, (len(filtered) + items_per_page - 1) // items_per_page)
    page = min(page, total_pages)
    start = (page - 1) * items_per_page
    end = min(start + items_per_page, len(filtered))
    st.info(f"Mostrando {start+1}–{end} de {len(filtered)} | Página {page}/{total_pages}")

    for i, rec in enumerate(filtered[start:end], start=start + 1):
        render_record(rec, i)
