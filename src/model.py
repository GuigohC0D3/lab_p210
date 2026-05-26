import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.config import MODEL_ID
from src.utils import vram_allocated_mb, reset_cuda_stats


def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(MODEL_ID)


def load_model(
    bnb_config: BitsAndBytesConfig,
    flash_attention: bool = False,
) -> tuple[AutoModelForCausalLM, float]:
    reset_cuda_stats()
    vram_before = vram_allocated_mb()

    kwargs = dict(
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    if flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)

    vram_used = vram_allocated_mb() - vram_before
    return model, vram_used


def build_input_ids(
    tokenizer: AutoTokenizer,
    contexto_rag: str,
    rag_slice: int,
    device,
):
    system_prompt = "Você é um assistente médico especializado. Gere um resumo clínico conciso e preciso."
    user_prompt = (
        f"Com base nos seguintes capítulos de manuais médicos recuperados:\n\n"
        f"{contexto_rag[:rag_slice]}\n\n"
        "Gere um resumo clínico de 500 palavras destacando os principais protocolos terapêuticos."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    print(f"Tokens de entrada (com template de chat): {input_ids.shape[1]:,}")
    return input_ids
