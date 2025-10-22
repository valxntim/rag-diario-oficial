import os
import json
import csv
import re

# Diretórios
RESULTS_DIR = "resultados_jsonl"
CSV_FILE = "resultados_finais_completo.csv"

def parse_filename(filename):
    """
    Extrai parâmetros do nome do arquivo.
    Formato esperado: {chunk}_{viz}_{k}_{base}_{modelo}.jsonl
    Exemplo: 400_0_1_a_llama3.jsonl
    """
    name = filename.replace('.jsonl', '')
    parts = name.split('_')
    
    if len(parts) >= 5:
        chunk = parts[0]
        viz = parts[1]
        k = parts[2]
        base = parts[3]
        modelo = '_'.join(parts[4:])  # caso o nome do modelo tenha underscore
        return chunk, viz, k, base, modelo
    return None

def processar_jsonl(filepath):
    """
    Processa um arquivo JSONL e retorna acertos, erros e total.
    """
    acertos = 0
    total = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                j = json.loads(line)
                total += 1
                if j.get('acerto', False):
                    acertos += 1
    except Exception as e:
        print(f"❌ Erro ao processar {filepath}: {e}")
        return 0, 0, 0
    
    erros = total - acertos
    return acertos, erros, total

def main():
    # Verifica se o diretório existe
    if not os.path.exists(RESULTS_DIR):
        print(f"❌ Diretório {RESULTS_DIR} não encontrado!")
        return
    
    # Lista todos os arquivos JSONL
    jsonl_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.jsonl')]
    
    if not jsonl_files:
        print(f"❌ Nenhum arquivo JSONL encontrado em {RESULTS_DIR}")
        return
    
    print(f"📁 Encontrados {len(jsonl_files)} arquivos JSONL")
    
    # Cria/sobrescreve o CSV
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['arquivo_jsonl', 'chunk', 'vizinho', 'k', 'modelo', 'acertos', 'erros', 'acuracia'])
        
        for filename in sorted(jsonl_files):
            filepath = os.path.join(RESULTS_DIR, filename)
            
            # Extrai parâmetros do nome
            params = parse_filename(filename)
            if not params:
                print(f"⚠️ Não foi possível parsear: {filename}")
                continue
            
            chunk, viz, k, base, modelo = params
            
            # Processa o arquivo
            acertos, erros, total = processar_jsonl(filepath)
            
            if total == 0:
                print(f"⚠️ Arquivo vazio ou com erro: {filename}")
                continue
            
            acuracia = (acertos / total * 100) if total else 0.0
            
            # Escreve no CSV
            arquivo_nome = filename.replace('.jsonl', '')
            w.writerow([arquivo_nome, chunk, viz, k, modelo, acertos, erros, f"{acuracia:.2f}%"])
            
            print(f"✅ {filename}: {acertos}/{total} ({acuracia:.2f}%)")
    
    print(f"\n🎉 CSV completo gerado: {CSV_FILE}")

if __name__ == "__main__":
    main()
