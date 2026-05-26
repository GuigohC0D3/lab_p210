import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def print_report(baseline: dict, optimized: dict, vram_model_mb: float) -> None:
    speedup = baseline["time"] / optimized["time"] if optimized["time"] > 0 else float("inf")
    reducao_vram = (
        (baseline["vram"] - optimized["vram"]) / baseline["vram"] * 100
        if baseline["vram"] > 0
        else 0
    )

    rows = [baseline, optimized]
    header = f"{'Configuração':<35} {'Tempo (s)':>10} {'Velocidade (tok/s)':>20} {'Pico VRAM (MB)':>15}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in rows:
        speed = r["tokens"] / r["time"] if r["time"] > 0 else 0
        print(f"{r['label']:<35} {r['time']:>10.2f} {speed:>20.2f} {r['vram']:>15.1f}")
    print("=" * len(header))
    print(f"\nSpeedup: {speedup:.2f}x mais rápido")
    print(f"Redução de VRAM: {reducao_vram:.1f}%")
    print(f"VRAM do modelo (4-bit QLoRA): {vram_model_mb:.1f} MB")


def plot_benchmark(baseline: dict, optimized: dict, output_path: str = "benchmark_results.png") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Lab 10 — Benchmark: Sem Cache vs KV Cache + FlashAttention-2",
        fontsize=13,
        fontweight="bold",
    )

    labels = [baseline["label"], optimized["label"]]
    cores = ["#e74c3c", "#2ecc71"]

    bars1 = axes[0].bar(labels, [baseline["time"], optimized["time"]], color=cores, edgecolor="black", width=0.5)
    axes[0].set_title("Tempo de Geração (100 tokens)", fontsize=11)
    axes[0].set_ylabel("Tempo (segundos)")
    for bar, val in zip(bars1, [baseline["time"], optimized["time"]]):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.2f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    bars2 = axes[1].bar(labels, [baseline["vram"], optimized["vram"]], color=cores, edgecolor="black", width=0.5)
    axes[1].set_title("Pico de VRAM durante Geração", fontsize=11)
    axes[1].set_ylabel("VRAM (MB)")
    for bar, val in zip(bars2, [baseline["vram"], optimized["vram"]]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{val:.0f} MB",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    red_patch = mpatches.Patch(color="#e74c3c", label="Sem otimização")
    green_patch = mpatches.Patch(color="#2ecc71", label="KV Cache + FlashAttn-2")
    fig.legend(handles=[red_patch, green_patch], loc="lower center", ncol=2, fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Gráfico salvo em {output_path}")
