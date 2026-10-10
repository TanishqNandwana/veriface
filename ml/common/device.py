"""
VeriFace Device Selection & Hardware Acceleration Helper
Supports Apple Silicon MPS, NVIDIA CUDA, and CPU fallback.
"""

import torch
import logging

logger = logging.getLogger("veriface.device")

def get_device(force_cpu: bool = False) -> torch.device:
    """
    Selects the optimal torch device available.
    Order of preference: MPS (Apple Silicon GPU) -> CUDA (NVIDIA GPU) -> CPU
    """
    if force_cpu:
        logger.info("Device manually forced to CPU.")
        return torch.device("cpu")
        
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Selected PyTorch MPS (Apple Silicon GPU) device.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info(f"Selected PyTorch CUDA device: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        logger.info("Selected CPU device (fallback).")
        return torch.device("cpu")

def get_device_info() -> dict:
    """
    Returns hardware device diagnostic details.
    """
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    has_cuda = torch.cuda.is_available()
    active_device = str(get_device())
    
    info = {
        "active_device": active_device,
        "mps_available": has_mps,
        "cuda_available": has_cuda,
        "cpu_threads": torch.get_num_threads(),
    }
    if has_cuda:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_count"] = torch.cuda.device_count()
    return info

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Device Info:", get_device_info())
