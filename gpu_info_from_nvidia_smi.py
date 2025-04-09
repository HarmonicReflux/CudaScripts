"""
This script queries NVIDIA GPU details using `nvidia-smi -q -x` and parses the XML output.
It reports selected GPU stats in a clean, readable format, including:

- Driver and CUDA version
- Product name and architecture
- Memory usage (used, free, total)
- GPU utilisation
- Core temperature and shutdown threshold

Supports multiple GPUs automatically.

Author: HarmonicReflux
"""


import subprocess
import xml.etree.ElementTree as ET

def get_text(parent, path, default="n/a", first_word_only=True):
    """Safely extract nested text from XML tree."""
    if parent is None:
        return default
    for tag in path:
        parent = parent.find(tag)
        if parent is None:
            return default
    if parent.text:
        return parent.text.strip().split()[0] if first_word_only else parent.text.strip()
    return default

def get_gpu_info():
    try:
        output = subprocess.check_output(["nvidia-smi", "-q", "-x"], encoding='utf-8')
    except subprocess.CalledProcessError as e:
        print("Failed to run nvidia-smi:", e)
        return []

    root = ET.fromstring(output)

    driver_version = root.findtext("driver_version", default="n/a")
    cuda_version = root.findtext("cuda_version", default="n/a")

    field_map = {
        "Product name": (["product_name"], False), 
        "Product architecture": (["product_architecture"], True),
        "Used GPU memory (MB)": (["fb_memory_usage", "used"], True),
        "Free GPU memory (MB)": (["fb_memory_usage", "free"], True),
        "Total GPU memory (MB)": (["fb_memory_usage", "total"], True),
        "Utilisation (%)": (["utilization", "gpu_util"], True),
        "GPU current temp (C)": (["temperature", "gpu_temp"], True),
        "GPU shutdown temp (C)": (["temperature", "gpu_temp_max_threshold"], True)
    }

    gpu_data = []

    for gpu in root.findall("gpu"):
        info = {
            "Driver version": driver_version,
            "CUDA version": cuda_version
        }
        for label, (path, first_word_only) in field_map.items():
            info[label] = get_text(gpu, path, first_word_only=first_word_only)
        gpu_data.append(info)

    return gpu_data


if __name__ == "__main__":
    gpu_info_list = get_gpu_info()
    for i, gpu_info in enumerate(gpu_info_list):
        print(f"\n--- GPU {i} ---")
        for key, value in gpu_info.items():
            print(f"{key}: {value}")
