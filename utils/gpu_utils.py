"""
GPU utilities for detecting and optimizing GPU usage.

This module provides functions to detect, query, and optimize GPU usage
for LLM inference across different hardware configurations.
"""

import os
import sys
import platform
import torch
from typing import Dict, Any, Optional

def torch_device():
    """
    Get the best available PyTorch device (CUDA, MPS, or CPU).
    
    Returns:
        torch.device: The device to use
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def is_gpu_available() -> bool:
    """
    Check if any GPU is available for use.
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    if torch.cuda.is_available():
        return True
    
    # Check for Apple Silicon GPU (MPS)
    if platform.system() == "Darwin" and hasattr(torch.backends, "mps"):
        return torch.backends.mps.is_available()
    
    return False

def get_gpu_info() -> Dict[str, Any]:
    """
    Get information about the available GPU.
    
    Returns:
        Dict[str, Any]: Dictionary with GPU information or CPU info if no GPU
    """
    # Initialize with basic info
    info = {
        "available": False,
        "name": "CPU (No GPU detected)",
        "memory_gb": 0,
        "platform": platform.system(),
        "type": "cpu"
    }
    
    # Check for CUDA (NVIDIA) GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            info["available"] = True
            info["type"] = "cuda"
            info["count"] = gpu_count
            
            # Get info for the first GPU (can be extended for multi-GPU)
            info["name"] = torch.cuda.get_device_name(0)
            info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return info
    
    # Check for Apple Silicon GPU (MPS)
    if platform.system() == "Darwin" and hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            info["available"] = True
            info["type"] = "mps"
            info["name"] = "Apple Silicon GPU"
            
            # Try to determine memory amount (not directly available through PyTorch)
            try:
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                system_memory = int(result.stdout.strip()) / (1024**3)
                
                # Approximate GPU memory as half of system memory (rough estimate)
                info["memory_gb"] = system_memory / 2
            except Exception:
                # Default to a conservative estimate
                info["memory_gb"] = 4
                
            return info
    
    return info

def get_device_string() -> str:
    """
    Get a human-readable string describing the current device.
    
    Returns:
        str: Description of the current device
    """
    info = get_gpu_info()
    
    if info["available"]:
        if info["type"] == "cuda":
            return f"NVIDIA GPU ({info['name']})"
        elif info["type"] == "mps":
            return f"Apple Silicon GPU ({info['memory_gb']:.1f} GB)"
    
    return "CPU (No GPU)"

def optimize_model_params(params: Dict[str, Any], provider: str) -> Dict[str, Any]:
    """
    Optimize model parameters based on available hardware and provider.
    
    Args:
        params (Dict[str, Any]): Original model parameters
        provider (str): Model provider (e.g., "llama.cpp", "ctransformers")
        
    Returns:
        Dict[str, Any]: Optimized parameters
    """
    gpu_info = get_gpu_info()
    optimized = params.copy()
    
    # GGUF providers (llama.cpp, ctransformers)
    if provider in ["llama.cpp", "ctransformers"]:
        if gpu_info["available"]:
            # Set n_gpu_layers based on GPU memory
            if "n_gpu_layers" in optimized:
                memory_gb = gpu_info["memory_gb"]
                
                if memory_gb < 4:
                    # Limited GPU memory - use fewer layers
                    optimized["n_gpu_layers"] = min(optimized["n_gpu_layers"], 20)
                elif memory_gb < 8:
                    # Medium GPU memory
                    optimized["n_gpu_layers"] = min(optimized["n_gpu_layers"], 35)
                else:
                    # Lots of GPU memory - use more layers
                    optimized["n_gpu_layers"] = 50
        else:
            # CPU only
            optimized["n_gpu_layers"] = 0
    
    # Transformers provider
    elif provider == "transformers":
        if gpu_info["available"]:
            # Use device_map="auto" for GPUs
            optimized["device_map"] = "auto"
            
            # Set quantization based on GPU memory
            memory_gb = gpu_info["memory_gb"]
            
            if memory_gb < 8:
                # For limited GPU memory, use 4-bit quantization
                optimized["load_in_4bit"] = True
                optimized["load_in_8bit"] = False
            elif memory_gb < 16:
                # For medium GPU memory, use 8-bit quantization
                optimized["load_in_8bit"] = True
                optimized["load_in_4bit"] = False
        else:
            # CPU only
            optimized["device_map"] = None
            optimized["load_in_4bit"] = False
            optimized["load_in_8bit"] = False
    
    # Ollama provider
    elif provider == "ollama":
        # Ollama handles GPU acceleration internally
        pass
    
    return optimized

# If this file is run directly, show GPU information
if __name__ == "__main__":
    device = torch_device()
    print(f"PyTorch device: {device}")
    
    gpu_info = get_gpu_info()
    if gpu_info["available"]:
        print(f"GPU available: {gpu_info['name']}")
        print(f"GPU memory: {gpu_info['memory_gb']:.2f} GB")
        print(f"GPU type: {gpu_info['type']}")
    else:
        print("No GPU available - using CPU")
    
    print(f"Device summary: {get_device_string()}")