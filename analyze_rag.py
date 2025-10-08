import json
import re
from collections import defaultdict

# ===== CHANGE THIS FILENAME =====
INPUT_FILE = "600_0_1_5_results_baseB.jsonl"    # your original JSONL
OUTPUT_FILE = "final_87_results.jsonl" # 90-item summary

def analyze_rag_evaluation(input_file, output_file):
    questions = defaultdict(list)
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            base_id = re.sub(r"_v\d+$", "", obj["id_versao_pergunta"])
            questions[base_id].append(obj)

    results = []
    correct_count = 0

    for base_id, versions in questions.items():
        correct = [v for v in versions if v.get("acerto")]
        if correct:
            correct_count += 1
            chosen = correct[0]
            status = "correct"
        else:
            chosen = versions[0]
            status = "wrong"

        results.append({
            "base_question_id":    base_id,
            "status":             status,
            "chosen_version":     chosen["id_versao_pergunta"],
            "pergunta":           chosen["pergunta"],
            "resposta_esperada":  chosen["resposta_esperada"],
            "resposta_gerada":    chosen["resposta_gerada"],
            "pdf":                chosen.get("pdf",""),
            "extrato":            chosen.get("extrato",""),
            "contextos_recuperados": chosen.get("contextos_recuperados", []),
            "total_versions":     len(versions),
            "correct_versions":   len(correct)
        })

    total = len(questions)
    acc = correct_count / total if total else 0

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=== RAG EVALUATION SUMMARY ===")
    print(f"Total questions: {total}")
    print(f"At least one correct: {correct_count}")
    print(f"All wrong: {total-correct_count}")
    print(f"Accuracy: {acc:.2%}")
    print(f"Saved to {output_file}")
    return results

if __name__=="__main__":
    analyze_rag_evaluation(INPUT_FILE, OUTPUT_FILE)
