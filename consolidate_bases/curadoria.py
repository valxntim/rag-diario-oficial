import json
import re

OUTPUT_FILE_A = "base_a_enhanced.jsonl"
OUTPUT_FILE_B = "base_b_normalized.jsonl"

def strip_base_id(s):
    return re.sub(r'(_v\d+)?$', '', s)

def collect_versions(filename):
    versions = {}
    with open(filename, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            full_id = obj["id_versao_pergunta"]
            base_id = strip_base_id(full_id)
            ver = full_id.split('_')[-1]
            if base_id not in versions:
                versions[base_id] = set()
            versions[base_id].add(ver)
    return versions

# Check for A
versions_A = collect_versions(OUTPUT_FILE_A)
versions_B = collect_versions(OUTPUT_FILE_B)

incomplete_A = [bid for bid, vers in versions_A.items() if set(vers) != {"v0", "v1", "v2"}]
incomplete_B = [bid for bid, vers in versions_B.items() if set(vers) != {"v0", "v1", "v2"}]

print("Base A version completeness:")
if not incomplete_A:
    print("✔ Every base ID in A has v0, v1, v2")
else:
    print("❌ These base IDs in A are missing versions:", incomplete_A)

print("\nBase B version completeness:")
if not incomplete_B:
    print("✔ Every base ID in B has v0, v1, v2")
else:
    print("❌ These base IDs in B are missing versions:", incomplete_B)
