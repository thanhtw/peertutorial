"""
Model downloader and manager for handling both Ollama and GGUF models.

This module provides utilities for downloading, checking, and managing LLM models
in both Ollama and GGUF formats.
"""

import os
import sys
import platform
import subprocess
import time
import streamlit as st
import requests
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Define constants for model sources
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = "llama3:1b"
MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

# Recommended models with download links
RECOMMENDED_MODELS = {
    "llama-3-8b-instruct.Q4_K_M.gguf": {
        "description": "Llama 3 8B Instruct (4-bit)",
        "url": "https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8b-instruct.Q4_K_M.gguf",
        "size_gb": 4.5,
        "type": "gguf",
        "family": "llama"
    },
    "deepseek-coder-6.7b-instruct.Q4_K_M.gguf": {
        "description": "DeepSeek Coder 6.7B (4-bit)",
        "url": "https://huggingface.co/TheBloke/deepseek-coder-6.7b-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
        "size_gb": 4.0,
        "type": "gguf",
        "family": "deepseek"
    },
    "phi-3-mini-4k-instruct.Q4_K_M.gguf": {
        "description": "Phi-3 Mini 4K (4-bit)",
        "url": "https://huggingface.co/TheBloke/phi-3-mini-4k-instruct-GGUF/resolve/main/phi-3-mini-4k-instruct.Q4_K_M.gguf",
        "size_gb": 1.8,
        "type": "gguf",
        "family": "phi"
    },
    "gemma-2b-it.Q4_K_M.gguf": {
        "description": "Gemma 2B Instruct (4-bit)",
        "url": "https://huggingface.co/TheBloke/Gemma-2B-it-GGUF/resolve/main/gemma-2b-it.Q4_K_M.gguf",
        "size_gb": 1.5,
        "type": "gguf",
        "family": "gemma"
    }
}

# Ensure model directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "gguf"), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "ollama"), exist_ok=True)

def check_ollama_running() -> bool:
    """
    Check if Ollama is running locally.
    
    Returns:
        bool: True if Ollama is running, False otherwise
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False

def start_ollama() -> bool:
    """
    Start Ollama locally based on the platform.
    
    Returns:
        bool: True if started successfully, False otherwise
    """
    system = platform.system()
    
    try:
        if system == "Windows":
            # For Windows
            proc = subprocess.Popen(
                ["ollama", "serve"], 
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
        elif system == "Linux":
            # Try systemd first on Linux
            try:
                subprocess.run(
                    ["systemctl", "--user", "start", "ollama"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5
                )
            except (subprocess.SubprocessError, FileNotFoundError):
                # Fallback to direct startup
                proc = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
        elif system == "Darwin":  # macOS
            proc = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
        else:
            print(f"Unsupported platform: {system}")
            return False
            
        # Wait for Ollama to start
        for i in range(10):  # Try for 10 seconds
            print(f"Starting Ollama... ({i+1}/10)")
            time.sleep(1)
            if check_ollama_running():
                print("Ollama started successfully!")
                return True
                
        print("Ollama did not start within expected time")
        return False
        
    except Exception as e:
        print(f"Error starting Ollama: {str(e)}")
        return False

def get_ollama_models() -> List[Dict[str, Any]]:
    """
    Get a list of all available Ollama models with additional information.
    
    Returns:
        list: List of model information dictionaries
    """
    # Default library of models that can be pulled
    library_models = [
        {"id": "llama3:1b", "name": "Llama 3 (1B)", "description": "Meta's Llama 3 1B model", "pulled": False, "tags": ["chat", "fast"]},
        {"id": "llama3", "name": "Llama 3", "description": "Meta's Llama 3 model", "pulled": False, "tags": ["chat", "general"]},
        {"id": "llama3:8b", "name": "Llama 3 (8B)", "description": "Meta's Llama 3 8B model", "pulled": False, "tags": ["chat", "general"]},
        {"id": "deepseek-coder:6.7b", "name": "DeepSeek Coder (6.7B)", "description": "Code-specialized model", "pulled": False, "tags": ["code", "technical"]},
        {"id": "phi3:mini", "name": "Phi-3 Mini", "description": "Microsoft Phi-3 model", "pulled": False, "tags": ["chat", "fast"]},
        {"id": "gemma:2b", "name": "Gemma 2B", "description": "Google's lightweight Gemma model", "pulled": False, "tags": ["chat", "fast"]},
        {"id": "codellama:7b", "name": "Code Llama 7B", "description": "Code-specialized Llama model", "pulled": False, "tags": ["code", "technical"]}
    ]
    
    # Check if Ollama is running
    if not check_ollama_running():
        # Return library models without checking if they're pulled
        return sorted(library_models, key=lambda x: x["name"])
    
    try:
        # Try to get list of already pulled models from Ollama
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        
        if response.status_code == 200:
            pulled_models = response.json().get("models", [])
            pulled_ids = [model["name"] for model in pulled_models]
            
            # Mark models as pulled if they exist
            for model in library_models:
                if model["id"] in pulled_ids:
                    model["pulled"] = True
            
            # Add any pulled models that aren't in our standard list
            for pulled_model in pulled_models:
                model_id = pulled_model["name"]
                if not any(model["id"] == model_id for model in library_models):
                    # Extract size information
                    size_str = pulled_model.get("size", "Unknown")
                    # Try to determine appropriate tags based on model name
                    tags = []
                    if "code" in model_id.lower():
                        tags.append("code")
                    if any(name in model_id.lower() for name in ["mini", "small", "tiny", "1b", "2b"]):
                        tags.append("fast")
                    else:
                        tags.append("general")
                    
                    # Add to the list
                    library_models.append({
                        "id": model_id,
                        "name": model_id,
                        "description": f"Size: {size_str}",
                        "pulled": True,
                        "tags": tags
                    })
        
        # Sort models: pulled models first, then by name
        return sorted(library_models, key=lambda x: (not x["pulled"], x["name"]))
    
    except Exception as e:
        print(f"Error getting Ollama models: {str(e)}")
        # Return unsorted library models if we can't connect
        return library_models

def pull_ollama_model(model_name: str) -> bool:
    """
    Pull an Ollama model.
    
    Args:
        model_name (str): Name of the model to pull
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Make sure Ollama is running
    if not check_ollama_running():
        print("Ollama is not running. Starting Ollama...")
        if not start_ollama():
            print("Failed to start Ollama.")
            return False
    
    try:
        print(f"Starting download of {model_name}...")
        
        # Start the pull operation
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": model_name}
        )
        
        if response.status_code != 200:
            print(f"Failed to start model download: {response.text}")
            return False
        
        # Monitor progress
        start_time = time.time()
        last_update_time = start_time
        
        pull_completed = False
        max_wait_time = 1800  # 30 minute timeout
        while not pull_completed and (time.time() - start_time) < max_wait_time:
            try:
                # Check if model exists in list of models
                check_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
                if check_response.status_code == 200:
                    models = check_response.json().get("models", [])
                    if any(model["name"] == model_name for model in models):
                        print(f"Model {model_name} downloaded successfully!")
                        pull_completed = True
                        break
                
                # Update progress message every 5 seconds
                current_time = time.time()
                if current_time - last_update_time >= 5:
                    elapsed_mins = (current_time - start_time) / 60
                    print(f"Downloading {model_name}... {elapsed_mins:.1f} minutes elapsed")
                    last_update_time = current_time
                
                time.sleep(1)
                
            except Exception as e:
                print(f"Error checking model status: {str(e)}")
                time.sleep(5)
        
        if not pull_completed:
            print(f"Download timeout for {model_name}. Model may still be downloading.")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error pulling model: {str(e)}")
        return False

def download_model_with_streamlit(model_info: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """
    Download a model with Streamlit progress indicators.
    
    Args:
        model_info (Dict[str, Dict[str, Any]]): Model information including URL
        
    Returns:
        Optional[str]: Path to downloaded model or None if failed
    """
    if not model_info:
        st.error("No model information provided.")
        return None
    
    try:
        # Get the first model name and info
        model_name = list(model_info.keys())[0]
        info = model_info[model_name]
        
        model_url = info.get("url")
        if not model_url:
            st.error(f"No download URL for model: {model_name}")
            return None
        
        # Determine the output path
        family = info.get("family", "").lower()
        model_type = info.get("type", "gguf").lower()
        
        if model_type == "gguf":
            output_dir = os.path.join(MODELS_DIR, "gguf")
        else:
            output_dir = os.path.join(MODELS_DIR, model_type)
            
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, model_name)
        
        # Check if file already exists
        if os.path.exists(output_path):
            st.info(f"Model {model_name} already exists at {output_path}")
            return output_path
        
        # Create progress bar and status message
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.info(f"Downloading {model_name}...")
        
        # Calculate total size
        size_gb = info.get("size_gb", 0)
        if size_gb:
            status_text.info(f"Downloading {model_name} ({size_gb:.1f} GB)...")
        
        # Download with progress
        with requests.get(model_url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            
            # Update expected size if it wasn't provided
            if not size_gb and total_size:
                size_gb = total_size / (1024**3)
                status_text.info(f"Downloading {model_name} ({size_gb:.1f} GB)...")
            
            # Download in chunks with progress updates
            chunk_size = 1024 * 1024  # 1MB chunks
            downloaded = 0
            
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update progress
                        if total_size:
                            progress = min(1.0, downloaded / total_size)
                            progress_bar.progress(progress)
                            
                            # Update status text with download speed and ETA
                            if downloaded > chunk_size:  # Skip first chunk for more accurate measurements
                                downloaded_mb = downloaded / (1024**2)
                                total_mb = total_size / (1024**2)
                                status_text.info(f"Downloading {model_name}: {downloaded_mb:.1f} MB / {total_mb:.1f} MB ({progress*100:.1f}%)")
        
        # Final progress update
        progress_bar.progress(1.0)
        status_text.success(f"Download complete: {model_name}")
        
        return output_path
        
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        return None

def download_model_with_wget(model_info: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """
    Download a model using wget in the terminal (fallback method).
    
    Args:
        model_info (Dict[str, Dict[str, Any]]): Model information including URL
        
    Returns:
        Optional[str]: Path to downloaded model or None if failed
    """
    if not model_info:
        print("No model information provided.")
        return None
    
    try:
        # Get the first model name and info
        model_name = list(model_info.keys())[0]
        info = model_info[model_name]
        
        model_url = info.get("url")
        if not model_url:
            print(f"No download URL for model: {model_name}")
            return None
        
        # Determine the output path
        family = info.get("family", "").lower()
        model_type = info.get("type", "gguf").lower()
        
        if model_type == "gguf":
            output_dir = os.path.join(MODELS_DIR, "gguf")
        else:
            output_dir = os.path.join(MODELS_DIR, model_type)
            
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, model_name)
        
        # Check if file already exists
        if os.path.exists(output_path):
            print(f"Model {model_name} already exists at {output_path}")
            return output_path
        
        # Download with wget
        print(f"Downloading {model_name} to {output_path}...")
        
        # Try wget first (Linux/Mac)
        try:
            subprocess.run(
                ["wget", "-O", output_path, model_url],
                check=True
            )
            print(f"Download complete: {model_name}")
            return output_path
        except (subprocess.SubprocessError, FileNotFoundError):
            # Try curl as fallback
            try:
                subprocess.run(
                    ["curl", "-L", "-o", output_path, model_url],
                    check=True
                )
                print(f"Download complete: {model_name}")
                return output_path
            except (subprocess.SubprocessError, FileNotFoundError):
                print("Neither wget nor curl is available. Please download manually.")
                return None
        
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return None

def get_installed_gguf_models() -> List[Dict[str, Any]]:
    """
    List all installed GGUF models.
    
    Returns:
        List[Dict[str, Any]]: List of model information
    """
    gguf_dir = os.path.join(MODELS_DIR, "gguf")
    models = []
    
    try:
        if os.path.exists(gguf_dir):
            for filename in os.listdir(gguf_dir):
                if filename.endswith(".gguf"):
                    file_path = os.path.join(gguf_dir, filename)
                    file_size = os.path.getsize(file_path) / (1024**3)  # Size in GB
                    
                    # Try to determine model family from filename
                    family = "unknown"
                    if "llama" in filename.lower():
                        family = "llama"
                    elif "mistral" in filename.lower():
                        family = "mistral"
                    elif "phi" in filename.lower():
                        family = "phi"
                    elif "gemma" in filename.lower():
                        family = "gemma"
                    elif "deepseek" in filename.lower():
                        family = "deepseek"
                    
                    # Try to determine quantization
                    quant = "unknown"
                    if "Q4_K_M" in filename:
                        quant = "Q4_K_M"
                    elif "Q5_K_M" in filename:
                        quant = "Q5_K_M"
                    elif "Q8_0" in filename:
                        quant = "Q8_0"
                    
                    models.append({
                        "id": file_path,
                        "name": filename,
                        "description": f"{family.capitalize()} model ({quant})",
                        "type": "gguf",
                        "family": family,
                        "size_gb": file_size,
                        "quantization": quant,
                        "path": file_path
                    })
    except Exception as e:
        print(f"Error listing GGUF models: {str(e)}")
    
    return models

# If this file is run directly, perform a basic test
if __name__ == "__main__":
    if check_ollama_running():
        print("Ollama is running")
        models = get_ollama_models()
        print(f"Found {len(models)} models:")
        for model in models:
            pulled_status = "✓" if model.get("pulled", False) else " "
            print(f"[{pulled_status}] {model['name']}: {model['description']}")
    else:
        print("Ollama is not running. Starting...")
        if start_ollama():
            print("Ollama started successfully")
        else:
            print("Failed to start Ollama")