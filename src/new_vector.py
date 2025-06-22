import fitz
import os
import torch
import io
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoProcessor
from huggingface_hub import snapshot_download
from .config import PDF_DIRECTORY

def teste_definitivo_nougat():
    print("--- Iniciando teste DEFINITIVO da pipeline Nougat ---")
    MODEL_TAG = "facebook/nougat-base"
    print(f"\n[ETAPA 1] Verificando/Baixando arquivos de '{MODEL_TAG}'...")
    local_model_path = snapshot_download(MODEL_TAG)
    print(f"[SUCESSO] Arquivos do modelo estão em: {local_model_path}")

    print(f"\n[ETAPA 2] Carregando o modelo e o processador universal...")
    model = VisionEncoderDecoderModel.from_pretrained(local_model_path)
    processor = AutoProcessor.from_pretrained(local_model_path)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    print("[SUCESSO] Modelo e processador carregados!")

    print("\n[ETAPA 3] Preparando e analisando o PDF...")
    pdf_filename = None
    for fname in os.listdir(PDF_DIRECTORY):
        if fname.lower().endswith(".pdf"):
            pdf_filename = fname
            break
    if not pdf_filename:
        print("ERRO: Nenhum PDF encontrado para o teste.")
        return

    print(f"Analisando a primeira página de: {pdf_filename}")
    doc = fitz.open(os.path.join(PDF_DIRECTORY, pdf_filename))
    page = doc[0]
    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")

    print("Pré-processando a imagem...")
    pixel_values = processor(images=image, padding=True, return_tensors="pt").pixel_values

    print("Analisando a imagem com Nougat (gerando a saída)...")
    outputs = model.generate(
        pixel_values.to(model.device),
        min_length=1,
        max_length=model.config.max_length,
        use_cache=True,
        do_sample=False,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.pad_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "").replace(processor.tokenizer.bos_token, "")

    print("\n\n--- SUCESSO FINAL! RESULTADO DA ANÁLISE (MARKDOWN) ---")
    print(sequence)
    print("--------------------------------------------------")

print("Script de teste definitivo iniciado...")
teste_definitivo_nougat()
print("Script de teste definitivo finalizado.")