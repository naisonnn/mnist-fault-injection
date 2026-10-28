# Automatisches Setup der Umgebung für das Projekt
# Installiert die passende PyTorch-Variante (CUDA, falls eine NVIDIA-GPU erkannt wird, ansonsten CPU)

import subprocess
import sys

# Erkennung der NVIDIA_GPU mithilfe von System-Befehl nvidia-smi
def detect_nvidia_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()

    except (FileNotFoundError, subprocess.SubprocessError):
        return ""

# Installiert die passende PyTorch-Variante basierend auf der erkannten GPU
def install_pytorch(backend):
    build = "cu126" if backend == "cuda" else "cpu"

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        f"torch==2.9.1+{build}",
        f"torchvision==0.24.1+{build}",
        "--index-url",
        f"https://download.pytorch.org/whl/{build}",
    ]

    print(f"Installiere PyTorch-Variante: {build}")
    subprocess.run(command, check=True)


gpu_name = detect_nvidia_gpu()
backend = "cuda" if gpu_name else "cpu"

print("NVIDIA-Grafikkarte:", gpu_name or "Keine erkannt")
print("Vorgesehene PyTorch-Variante:", backend)

install_pytorch(backend)