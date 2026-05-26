import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import MAX_NEW_TOKENS
from src.utils import reset_cuda_stats, cuda_sync, vram_peak_mb


def _generate(model, input_ids, use_cache: bool) -> tuple[torch.Tensor, float]:
    reset_cuda_stats()
    cuda_sync()
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=use_cache,
        )
    cuda_sync()
    return output, time.time() - t0


def run_without_cache(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
) -> dict:
    model.config.use_cache = False
    output, elapsed = _generate(model, input_ids, use_cache=False)

    n_generated = output.shape[1] - input_ids.shape[1]
    text = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    peak_vram = vram_peak_mb()

    print(f"\n=== MÉTRICA PASSO 3 — SEM KV CACHE ===")
    print(f"Tokens gerados: {n_generated}")
    print(f"Tempo total de geração: {elapsed:.2f}s")
    print(f"Velocidade: {n_generated / elapsed:.2f} tokens/s")
    print(f"Pico de VRAM: {peak_vram:.1f} MB")
    print(f"\nTexto gerado (primeiros 300 chars):\n{text[:300]}")

    return {"label": "Sem KV Cache (baseline)", "time": elapsed, "tokens": n_generated, "vram": peak_vram}


def run_with_cache(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
) -> dict:
    model.config.use_cache = True
    output, elapsed = _generate(model, input_ids, use_cache=True)

    n_generated = output.shape[1] - input_ids.shape[1]
    text = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    peak_vram = vram_peak_mb()

    print(f"\n=== MÉTRICA PASSO 4 — COM KV CACHE + FLASHATTENTION-2 ===")
    print(f"Tokens gerados: {n_generated}")
    print(f"Tempo total de geração: {elapsed:.2f}s")
    print(f"Velocidade: {n_generated / elapsed:.2f} tokens/s")
    print(f"Pico de VRAM: {peak_vram:.1f} MB")
    print(f"\nTexto gerado (primeiros 300 chars):\n{text[:300]}")

    return {"label": "KV Cache + FlashAttention-2", "time": elapsed, "tokens": n_generated, "vram": peak_vram}
