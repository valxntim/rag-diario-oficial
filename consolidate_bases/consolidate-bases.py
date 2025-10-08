import json
import csv
import re
from collections import defaultdict

BASE_A_FILE = "dataset_verificado_final_100_match.jsonl"
BASE_B_FILE = "benchmark_final_valor.jsonl"
CSV_FILE = "auditoria_dataset_final_100_match.csv"
ENRICH_FILE = "dataset_enriquecido.jsonl"

OUT_A = "base_a_enhanced.jsonl"
OUT_B = "base_b_normalized.jsonl"
REPORT_FILE = "consolidation_report.json"

def strip_all_versions(s):
    return re.sub(r'(_v\d+)*(_Q[1-3])?$', '', s)

def collect_enriched_fields():
    enriq = {}
    with open(ENRICH_FILE, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            key = strip_all_versions(obj["id"])
            enriq[key] = {
                "pergunta": obj.get("question", obj.get("pergunta", "")),
                "objeto": obj.get("objeto", ""),
                "source_text": obj.get("source_text", "")
            }
    return enriq

def collect_baseA():
    baseA = {}
    with open(BASE_A_FILE, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            id_field = obj.get("id_versao_pergunta") or obj.get("id")
            key = strip_all_versions(id_field)
            version = id_field[-2:]
            baseA[(key, version)] = obj
    return baseA

def collect_baseB():
    baseB = {}
    with open(BASE_B_FILE, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            id_field = obj["id_versao_pergunta"]
            key = strip_all_versions(id_field)
            version = id_field[-2:]
            baseB[(key, version)] = obj
    return baseB

def get_expected_ids(csv_filename):
    ids = set()
    csv_entries = {}
    with open(csv_filename, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        id_col = reader.fieldnames[0]
        for row in reader:
            if row["similarity_score"] == "100" and row["validado"].lower() == "true":
                key = strip_all_versions(row[id_col].strip())
                csv_entries[key] = {
                    "pdf": row.get("encontrado_em_pdf", ""),
                    "valor": row.get("valor", "")
                }
                for v in ["v0", "v1", "v2"]:
                    ids.add((key, v))
    return sorted(ids), csv_entries

enriq = collect_enriched_fields()
baseA = collect_baseA()
baseB = collect_baseB()
expected_ids, csv_entries = get_expected_ids(CSV_FILE)

out_a_ids = set()
out_b_ids = set()

# ----------- BASE A OUTPUT -------------
with open(OUT_A, "w", encoding="utf-8") as fa:
    for (baseid, ver) in expected_ids:
        a_rec = baseA.get((baseid, ver), {})
        id_full = f"{baseid}_{ver}"

        pergunta = (a_rec.get("pergunta", "") or 
                    a_rec.get("question", "") or 
                    enriq.get(baseid, {}).get("pergunta", ""))
        objeto = (a_rec.get("objeto", "") or 
                  enriq.get(baseid, {}).get("objeto", ""))

        resposta = (a_rec.get("resposta") or 
                    a_rec.get("answer") or 
                    csv_entries[baseid].get("valor", ""))

        # pdf from CSV ONLY; extrato always from enrichment's source_text
        pdf = csv_entries[baseid].get("pdf", "")
        extrato = enriq.get(baseid, {}).get("source_text", "")

        outa = {
            "id_versao_pergunta": id_full,
            "pergunta": pergunta,
            "objeto": objeto,
            "resposta": resposta,
            "pdf": pdf,
            "extrato": extrato,
        }
        fa.write(json.dumps(outa, ensure_ascii=False) + "\n")
        out_a_ids.add(id_full)

# ----------- BASE B OUTPUT -------------
version_map = {"_Q1": "v0", "_Q2": "v1", "_Q3": "v2"}
with open(OUT_B, "w", encoding="utf-8") as fb:
    for ((baseid, _), rec) in baseB.items():
        orig = rec["id_versao_pergunta"]
        ver = None
        for q, v in version_map.items():
            if orig.endswith(q):
                ver = v
                break
        if ver is None:
            continue
        id_full = f"{baseid}_{ver}"
        # pdf from CSV if possible; extrato from enriched
        pdf = csv_entries.get(baseid, {}).get("pdf", rec.get("pdf", ""))
        extrato = enriq.get(baseid, {}).get("source_text", "")

        outb = {
            "id_versao_pergunta": id_full,
            "pergunta": rec.get("pergunta", ""),
            "objeto": enriq.get(baseid, {}).get("objeto", ""),
            "resposta": rec.get("resposta", ""),
            "pdf": pdf,
            "extrato": extrato,
        }
        fb.write(json.dumps(outb, ensure_ascii=False) + "\n")
        out_b_ids.add(id_full)

# ----------- VALIDATION & DIFF REPORT -------------
missing_in_B = sorted(out_a_ids - out_b_ids)
missing_in_A = sorted(out_b_ids - out_a_ids)

with open(REPORT_FILE, "w", encoding="utf-8") as rf:
    json.dump({
        "A_not_in_B": missing_in_B,
        "B_not_in_A": missing_in_A,
        "A_total": len(out_a_ids),
        "B_total": len(out_b_ids),
        "ids_all_match": (missing_in_B == [] and missing_in_A == [])
    }, rf, indent=2, ensure_ascii=False)

print(f"✔ Output: Base A = {len(out_a_ids)}, Base B = {len(out_b_ids)}")
if not missing_in_B and not missing_in_A:
    print("✔ All IDs in Base A also in Base B")
else:
    print(f"❌ Mismatches: {len(missing_in_B)} A not in B, {len(missing_in_A)} B not in A")
    print("Report saved for diff investigation:", REPORT_FILE)
