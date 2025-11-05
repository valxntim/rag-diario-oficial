import pandas as pd
import sys
import json
import re
import os
import csv

# ============================================================================
# SCRIPT DE GERAÇÃO DE RELATÓRIO FINAL (BASE PEQUENA)
# ============================================================================

# 1. A pasta que contém TODOS os 94 arquivos .jsonl (os 58 válidos + 36 novos)
PASTA_RESULTADOS_COMPLETA = "resultados_valido_base_menor"

# 2. O nome do arquivo CSV final que queremos criar
CSV_SAIDA_FINAL = "relatorio_final_base_pequena_COMPLETO.csv"
# ============================================================================

# --- Função de Leitura Robusta (do script de auditoria) ---
def load_jsonl_robustly_audit(file_path: str):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if not line.strip(): continue
                try: data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"  > ⚠️ Aviso: Linha corrompida (Linha {i+1}) em {file_path}.")
                    return None
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"  > ❌ ERRO: Arquivo não encontrado: {file_path}"); return None
    except Exception as e:
        print(f"  > ❌ ERRO ao ler arquivo: {e}"); return None

# --- Script Principal ---
print(f"--- Gerando Relatório Final da Base Pequena ---")
print(f"Lendo todos os .jsonl de: {PASTA_RESULTADOS_COMPLETA}")

# Lista para armazenar os resultados de CADA arquivo
resultados_finais = []

if not os.path.exists(PASTA_RESULTADOS_COMPLETA):
    print(f"❌ ERRO: Pasta de resultados '{PASTA_RESULTADOS_COMPLETA}' não encontrada.")
    sys.exit()

# Itera por todos os arquivos na pasta
for filename in os.listdir(PASTA_RESULTADOS_COMPLETA):
    if not filename.endswith(".jsonl"):
        continue

    filepath = os.path.join(PASTA_RESULTADOS_COMPLETA, filename)
    
    # 1. Parsear o nome do arquivo para obter os parâmetros
    match = re.match(r'(\d+)_(\d+)_(\d+)_([ab])_(\w+)\.jsonl', filename)
    if not match:
        print(f"Aviso: Pulando arquivo com nome inválido: {filename}")
        continue
    
    chunk, viz, k, base_name, model_name = match.groups()
    
    # 2. Ler o arquivo .jsonl
    df = load_jsonl_robustly_audit(filepath)
    if df is None or df.empty:
        print(f"Aviso: Pulando arquivo vazio ou corrompido: {filename}")
        continue

    # 3. Calcular as estatísticas
    total = len(df)
    acertos = df['acerto'].sum() # .sum() em um booleano conta os True
    erros = total - acertos
    acuracia = (acertos / total * 100) if total > 0 else 0.0
    
    # 4. Armazena o resultado
    resultados_finais.append({
        'arquivo_jsonl': filename.replace('.jsonl', ''),
        'chunk': int(chunk),
        'vizinho': int(viz),
        'k': int(k),
        'modelo': model_name,
        'base': base_name,
        'acertos': acertos,
        'erros': erros,
        'acuracia': f"{acuracia:.2f}%"
    })

# 5. Salva o CSV Final
if not resultados_finais:
    print("❌ Nenhum arquivo de resultado válido foi encontrado.")
else:
    # Converte a lista de resultados em um DataFrame
    df_final = pd.DataFrame(resultados_finais)
    # Ordena o relatório (opcional, mas bom para visualização)
    df_final.sort_values(by=['chunk', 'vizinho', 'k', 'base'], inplace=True)
    
    # Salva no CSV
    df_final.to_csv(CSV_SAIDA_FINAL, index=False, encoding='utf-8-sig')
    print(f"\n✅ SUCESSO! Relatório final com {len(df_final)} linhas salvo como: '{CSV_SAIDA_FINAL}'")
    print("\n--- Amostra do Relatório ---")
    print(df_final.head())