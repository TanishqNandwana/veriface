#!/usr/bin/env python3
"""
VeriFace Development Environment Check Script
Detects system architecture, Python version, PyTorch devices (MPS/CPU),
audio/video/image libraries, backend components, and external CLI tools.
"""

import os
import sys
import platform
import subprocess
import shutil

def check_python():
    print(f"[+] Python Version: {sys.version.split()[0]} ({platform.platform()})")
    assert sys.version_info >= (3, 8), "Python 3.8+ required."

def check_pytorch():
    try:
        import torch
        print(f"[+] PyTorch Version: {torch.__version__}")
        
        # Check MPS (Apple Silicon)
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        has_cuda = torch.cuda.is_available()
        
        if has_mps:
            device = torch.device("mps")
            print("    -> Acceleration: PyTorch MPS (Apple Silicon GPU) available!")
        elif has_cuda:
            device = torch.device("cuda")
            print(f"    -> Acceleration: PyTorch CUDA available ({torch.cuda.get_device_name(0)})")
        else:
            device = torch.device("cpu")
            print("    -> Acceleration: CPU Fallback (No MPS/CUDA detected)")
            
        # Test basic tensor creation
        t = torch.ones((2, 2), device=device)
        print(f"    -> Tensor creation on '{device}' successful: sum={t.sum().item()}")
        return device
    except ImportError:
        print("[!] PyTorch is NOT installed!")
        return None

def check_vision():
    results = {}
    try:
        import cv2
        print(f"[+] OpenCV Version: {cv2.__version__}")
        results["opencv"] = True
    except ImportError:
        print("[!] OpenCV is NOT installed.")
        results["opencv"] = False

    try:
        import PIL
        print(f"[+] Pillow Version: {PIL.__version__}")
        results["pillow"] = True
    except ImportError:
        print("[!] Pillow is NOT installed.")
        results["pillow"] = False

    try:
        import torchvision
        print(f"[+] Torchvision Version: {torchvision.__version__}")
        results["torchvision"] = True
    except ImportError:
        print("[!] Torchvision is NOT installed.")
        results["torchvision"] = False

    return results

def check_audio():
    results = {}
    try:
        import librosa
        print(f"[+] Librosa Version: {librosa.__version__}")
        results["librosa"] = True
    except ImportError:
        print("[!] Librosa is NOT installed.")
        results["librosa"] = False

    try:
        import torchaudio
        print(f"[+] Torchaudio Version: {torchaudio.__version__}")
        results["torchaudio"] = True
    except ImportError:
        print("[!] Torchaudio is NOT installed.")
        results["torchaudio"] = False

    try:
        import soundfile
        print(f"[+] SoundFile Version: {soundfile.__version__}")
        results["soundfile"] = True
    except ImportError:
        print("[!] SoundFile is NOT installed.")
        results["soundfile"] = False

    return results

def check_backend():
    results = {}
    for pkg in ["fastapi", "uvicorn", "pydantic", "sqlalchemy", "reportlab", "matplotlib", "scipy", "sklearn"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            print(f"[+] {pkg}: {ver}")
            results[pkg] = True
        except ImportError:
            print(f"[!] {pkg} is NOT installed.")
            results[pkg] = False
    return results

def check_tools():
    tools = ["ffmpeg", "node", "npm", "docker"]
    for t in tools:
        path = shutil.which(t)
        if path:
            print(f"[+] Tool '{t}': {path}")
        else:
            print(f"[!] Tool '{t}': NOT found in PATH")

def main():
    print("==================================================")
    print(" VeriFace Environment Check ")
    print("==================================================")
    check_python()
    device = check_pytorch()
    check_vision()
    check_audio()
    check_backend()
    check_tools()
    print("==================================================")

if __name__ == "__main__":
    main()
