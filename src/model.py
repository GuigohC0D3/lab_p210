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


def _resolve_attn_impl(flash_attention: bool) -> str | None:
    """Returns flash_attention_2 for Ampere+, sdpa for older GPUs, None if disabled."""
    if not flash_attention or not torch.cuda.is_available():
        return None
    major, _ = torch.cuda.get_device_capability()
    return "flash_attention_2" if major >= 8 else "sdpa"


def load_model(
    bnb_config: BitsAndBytesConfig,
    flash_attention: bool = False,
) -> tuple[AutoModelForCausalLM, float, str]:
    reset_cuda_stats()
    vram_before = vram_allocated_mb()

    attn_impl = _resolve_attn_impl(flash_attention)
    kwargs = dict(
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
        print(f"Attention implementation: {attn_impl}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)

    vram_used = vram_allocated_mb() - vram_before
    return model, vram_used, attn_impl or "eager"


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
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    token_list = tokenizer.encode(formatted)
    input_ids = torch.tensor([token_list], dtype=torch.long).to(device)
    print(f"Tokens de entrada (com template de chat): {input_ids.shape[1]:,}")
    return input_ids
