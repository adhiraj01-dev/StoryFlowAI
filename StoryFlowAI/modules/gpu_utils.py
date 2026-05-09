"""
gpu_utils.py — CPU/GPU auto-detection and optimization helpers.

Usage:
    from modules.gpu_utils import get_device, get_dtype, log_device_info
"""

import torch
import streamlit as st


def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon support
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """
    Return fp16 for CUDA (faster + less VRAM), fp32 for everything else.
    MPS has partial fp16 support — use fp32 to stay safe.
    """
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def get_gpu_memory_info() -> dict:
    """Return VRAM stats for the active CUDA device (empty dict if CPU/MPS)."""
    if not torch.cuda.is_available():
        return {}
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    free = total - reserved
    return {
        "name": props.name,
        "total_gb": round(total, 2),
        "reserved_gb": round(reserved, 2),
        "allocated_gb": round(allocated, 2),
        "free_gb": round(free, 2),
    }


def clear_gpu_cache():
    """Free unused CUDA memory — call after heavy inference."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_device_info():
    """
    Display a device status badge in the Streamlit sidebar.
    Call once from app.py after page config is set.
    """
    device = get_device()
    dtype = get_dtype(device)

    icons = {"cuda": "⚡", "mps": "🍎", "cpu": "🖥️"}
    labels = {"cuda": "GPU (CUDA)", "mps": "GPU (Apple MPS)", "cpu": "CPU"}
    icon = icons.get(device.type, "🖥️")
    label = labels.get(device.type, device.type.upper())

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{icon} Device:** `{label}`")
    st.sidebar.markdown(f"**Precision:** `{'fp16' if dtype == torch.float16 else 'fp32'}`")

    if device.type == "cuda":
        info = get_gpu_memory_info()
        if info:
            st.sidebar.markdown(f"**GPU:** `{info['name']}`")
            st.sidebar.markdown(
                f"**VRAM:** `{info['allocated_gb']} / {info['total_gb']} GB used`"
            )

    return device, dtype
