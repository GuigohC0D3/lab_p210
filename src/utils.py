import gc
import torch


def bytes_to_mb(b: int) -> float:
    return b / (1024 ** 2)


def reset_cuda_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def vram_allocated_mb() -> float:
    if torch.cuda.is_available():
        return bytes_to_mb(torch.cuda.memory_allocated())
    return 0.0


def vram_peak_mb() -> float:
    if torch.cuda.is_available():
        return bytes_to_mb(torch.cuda.max_memory_allocated())
    return 0.0


def print_device_info() -> None:
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM total: {bytes_to_mb(props.total_memory):.1f} MB")
