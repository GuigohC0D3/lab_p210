import gc

from src.config import RAG_CONTEXT_SLICE
from src.model import build_bnb_config, load_tokenizer, load_model, build_input_ids
from src.rag import build_rag_context
from src.benchmark import run_without_cache, run_with_cache
from src.utils import print_device_info, reset_cuda_stats
from src.visualize import print_report, plot_benchmark


def main() -> None:
    print_device_info()

    bnb_config = build_bnb_config()
    tokenizer = load_tokenizer()

    # Passo 1: carregar modelo com QLoRA 4-bit
    print("\n=== PASSO 1: Carregamento QLoRA 4-bits ===")
    model, vram_model_mb, _ = load_model(bnb_config, flash_attention=False)
    print(f"VRAM ocupada pelo modelo quantizado (4-bit): {vram_model_mb:.1f} MB")
    print(f"(Equivalente em float16 seria ~2.200 MB — redução de ~75%)")

    # Passo 2: construir contexto RAG
    print("\n=== PASSO 2: Simulando RAG massivo ===")
    contexto_rag, _ = build_rag_context(tokenizer)
    input_ids = build_input_ids(tokenizer, contexto_rag, RAG_CONTEXT_SLICE, model.device)

    # Passo 3: geração SEM KV Cache (baseline)
    print("\n=== PASSO 3: Geração SEM KV Cache ===")
    baseline = run_without_cache(model, tokenizer, input_ids)

    # Passo 4: recarregar com FlashAttention-2 e gerar COM KV Cache
    print("\n=== PASSO 4: Recarregando com FlashAttention-2 + KV Cache ===")
    del model
    reset_cuda_stats()
    gc.collect()

    model_opt, vram_opt, attn_used = load_model(bnb_config, flash_attention=True)
    print(f"Modelo otimizado carregado ({attn_used}). VRAM: {vram_opt:.1f} MB")
    optimized = run_with_cache(model_opt, tokenizer, input_ids, attn_label=attn_used)

    # Relatório final
    print("\n=== RELATÓRIO COMPARATIVO FINAL ===")
    print_report(baseline, optimized, vram_model_mb)
    plot_benchmark(baseline, optimized)


if __name__ == "__main__":
    main()
