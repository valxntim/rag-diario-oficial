import json
import requests
import re
import os
from typing import Dict, Set

# Configurações do Ollama
OLLAMA_LLM_URL = "http://localhost:11434/api/generate"
OLLAMA_LLM_MODEL = "llama3.1:8b-instruct-q4_K_M"

# Prompt otimizado para gerar UMA pergunta melhorada
PROMPT_MELHORAR_PERGUNTA = """
CONTEXTO: Esta pergunta FALHOU no sistema RAG. Gere UMA pergunta MELHOR.

EXTRATO DO CONTRATO:
{extrato}

PERGUNTA ORIGINAL QUE FALHOU:
{pergunta_original}

RESPOSTA ESPERADA:
{resposta_esperada}

INSTRUÇÕES:
1. Analise por que a pergunta original pode ter falhado
2. Use EXATAMENTE os termos que aparecem no extrato
3. Se há número de contrato no extrato, USE ele na pergunta
4. Se há nome de empresa no extrato, USE exatamente como está
5. Se há valor/objeto no extrato, seja específico sobre isso
6. EVITE termos ambíguos ou que não estejam claros no extrato
7. Seja DIRETO e ESPECÍFICO

Gere APENAS UMA pergunta melhorada (sem numeração, sem explicação):
"""

def gerar_pergunta_melhorada(extrato: str, pergunta_original: str, resposta_esperada: str) -> str:
    """
    Gera uma pergunta melhorada usando o LLM
    """
    prompt = PROMPT_MELHORAR_PERGUNTA.format(
        extrato=extrato,
        pergunta_original=pergunta_original,
        resposta_esperada=resposta_esperada
    )
    
    try:
        response = requests.post(
            OLLAMA_LLM_URL,
            json={
                "model": OLLAMA_LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,  # Baixa temperatura para consistência
                    "top_p": 0.8,
                    "num_predict": 100   # Limitar resposta
                }
            },
            timeout=100  # Timeout maior para llama4
        )
        
        if response.status_code == 200:
            result = response.json()["response"].strip()
            
            # Limpar a resposta (remover numeração se houver)
            result = re.sub(r'^\d+\.\s*', '', result)
            result = result.strip()
            
            # Garantir que termina com ?
            if not result.endswith('?'):
                result += '?'
                
            return result
        else:
            print(f"❌ Erro na API: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Erro gerando pergunta: {e}")
        return None

def carregar_ids_com_erro(arquivo_acertos_erros: str) -> Set[str]:
    """
    Carrega IDs que tiveram erro (acerto: false)
    """
    ids_com_erro = set()
    
    if not os.path.exists(arquivo_acertos_erros):
        print(f"❌ Arquivo {arquivo_acertos_erros} não encontrado!")
        return ids_com_erro
    
    with open(arquivo_acertos_erros, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # Verificar se teve erro
                if not data.get('acerto', True):  # acerto: false ou campo não existe
                    ids_com_erro.add(data['id_versao_pergunta'])
            except Exception as e:
                print(f"⚠️  Erro lendo linha: {e}")
                continue
    
    return ids_com_erro

def carregar_perguntas_originais(arquivo_perguntas: str) -> Dict[str, dict]:
    """
    Carrega todas as perguntas originais indexadas por ID
    """
    perguntas = {}
    
    if not os.path.exists(arquivo_perguntas):
        print(f"❌ Arquivo {arquivo_perguntas} não encontrado!")
        return perguntas
    
    with open(arquivo_perguntas, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                perguntas[data['id_versao_pergunta']] = data
            except Exception as e:
                print(f"⚠️  Erro lendo linha: {e}")
                continue
    
    return perguntas

def processar_melhorias(arquivo_acertos_erros: str, arquivo_perguntas: str, arquivo_saida: str):
    """
    Processo principal: identifica erros e gera perguntas melhoradas
    """
    print("=== SCRIPT DE MELHORIA DE PERGUNTAS RAG ===\n")
    
    # 1. Carregar IDs com erro
    print("🔍 Identificando IDs com erro...")
    ids_com_erro = carregar_ids_com_erro(arquivo_acertos_erros)
    print(f"❌ Encontrados {len(ids_com_erro)} IDs com erro")
    
    if not ids_com_erro:
        print("✅ Nenhum erro encontrado! Nada a fazer.")
        return
    
    # 2. Carregar perguntas originais
    print("📝 Carregando perguntas originais...")
    perguntas_originais = carregar_perguntas_originais(arquivo_perguntas)
    print(f"📊 Carregadas {len(perguntas_originais)} perguntas originais")
    
    # 3. Processar cada pergunta
    perguntas_finais = []
    melhorias_feitas = 0
    erros_processamento = 0
    
    for id_pergunta, dados_pergunta in perguntas_originais.items():
        if id_pergunta in ids_com_erro:
            # Este ID teve erro - precisa gerar pergunta melhorada
            print(f"🔄 Melhorando: {id_pergunta}")
            
            nova_pergunta = gerar_pergunta_melhorada(
                dados_pergunta['extrato'],
                dados_pergunta['pergunta'],
                dados_pergunta['resposta']
            )
            
            if nova_pergunta:
                # Manter todos os campos, apenas trocar a pergunta
                dados_melhorados = dados_pergunta.copy()
                dados_melhorados['pergunta'] = nova_pergunta
                perguntas_finais.append(dados_melhorados)
                melhorias_feitas += 1
                print(f"  ✅ Nova pergunta: {nova_pergunta[:80]}...")
            else:
                # Se falhou, manter pergunta original
                perguntas_finais.append(dados_pergunta)
                erros_processamento += 1
                print(f"  ❌ Falha na geração - mantendo original")
        else:
            # Este ID não teve erro - manter como está
            perguntas_finais.append(dados_pergunta)
    
    # 4. Salvar arquivo final
    print(f"\n💾 Salvando arquivo final...")
    with open(arquivo_saida, 'w', encoding='utf-8') as f:
        for pergunta in perguntas_finais:
            f.write(json.dumps(pergunta, ensure_ascii=False) + '\n')
    
    # 5. Estatísticas finais
    print(f"\n🎉 PROCESSO CONCLUÍDO!")
    print(f"📊 Total de perguntas: {len(perguntas_finais)}")
    print(f"✨ Melhorias feitas: {melhorias_feitas}")
    print(f"❌ Erros de processamento: {erros_processamento}")
    print(f"✅ Mantidas sem alteração: {len(perguntas_finais) - melhorias_feitas}")
    print(f"💾 Arquivo salvo: {arquivo_saida}")

def main():
    """
    Função principal
    """
    # Arquivos de entrada e saída
    arquivo_acertos_erros = "600_0_1_5_baseBAtualizada.jsonl"  # Arquivo com acerto: true/false
    arquivo_perguntas = "base_b_extrato_final.jsonl"      # Arquivo com perguntas originais
    arquivo_saida = "base_b_extrato_final2.jsonl"   # Arquivo de saída
    
    print("Configuração dos arquivos:")
    print(f"📥 Acertos/Erros: {arquivo_acertos_erros}")
    print(f"📥 Perguntas Originais: {arquivo_perguntas}")
    print(f"📤 Arquivo de Saída: {arquivo_saida}")
    print()
    
    # Verificar se arquivos existem
    if not os.path.exists(arquivo_acertos_erros):
        print(f"❌ Arquivo não encontrado: {arquivo_acertos_erros}")
        return
    
    if not os.path.exists(arquivo_perguntas):
        print(f"❌ Arquivo não encontrado: {arquivo_perguntas}")
        return
    
    # Processar melhorias
    processar_melhorias(arquivo_acertos_erros, arquivo_perguntas, arquivo_saida)

if __name__ == "__main__":
    main()
