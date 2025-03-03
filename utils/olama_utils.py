"""
Unified Ollama utilities for managing Ollama services and models.
Provides centralized functions for Ollama operations across the application.
"""

import os
import sys
import platform
import time
import subprocess
import requests
import shutil
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
OLLAMA_MODELS_DIR = os.path.join(MODELS_DIR, "ollama")
DEFAULT_OLLAMA_MODEL = "llama3:8b"

# Ensure model directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OLLAMA_MODELS_DIR, exist_ok=True)

def is_ollama_running() -> bool:
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

def is_docker_ollama_running() -> bool:
    """
    Check if Ollama is running in Docker.
    
    Returns:
        bool: True if Ollama Docker is running, False otherwise
    """
    try:
        # Check if Docker is available
        docker_check = subprocess.run(
            ["docker", "--version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            timeout=2
        )
        
        if docker_check.returncode != 0:
            return False
            
        # Check for running Ollama container
        container_check = subprocess.run(
            ["docker", "ps", "--filter", "name=ollama", "--format", "{{.Names}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=2
        )
        
        return container_check.returncode == 0 and container_check.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def start_local_ollama(position: str = "sidebar") -> bool:
    """
    Start Ollama locally based on the platform.
    
    Args:
        position (str): Where to show notifications ("sidebar" or "content")
    
    Returns:
        bool: True if started successfully, False otherwise
    """
    try:
        system = platform.system()
        status_placeholder = st.sidebar.empty() if position == "sidebar" else st.empty()
        status_placeholder.info(f"Starting Ollama on {system}...")
        
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
            status_placeholder.error(f"Unsupported platform: {system}")
            return False
            
        # Wait for Ollama to start
        for i in range(10):  # Try for 10 seconds
            status_placeholder.info(f"Starting Ollama... ({i+1}/10)")
            time.sleep(1)
            if is_ollama_running():
                status_placeholder.success("Ollama started successfully!")
                return True
                
        status_placeholder.warning("Ollama did not start within expected time. It might still be starting...")
        return False
        
    except Exception as e:
        if position == "sidebar":
            st.sidebar.error(f"Error starting Ollama: {str(e)}")
        else:
            st.error(f"Error starting Ollama: {str(e)}")
        return False

def start_docker_ollama(position: str = "sidebar") -> bool:
    """
    Start Ollama in Docker.
    
    Args:
        position (str): Where to show notifications ("sidebar" or "content")
    
    Returns:
        bool: True if started successfully, False otherwise
    """
    try:
        status_placeholder = st.sidebar.empty() if position == "sidebar" else st.empty()
        status_placeholder.info("Starting Ollama Docker container...")
        
        # Check if Docker is installed
        docker_check = subprocess.run(
            ["docker", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if docker_check.returncode != 0:
            status_placeholder.error("Docker is not installed or not in PATH. Please install Docker first.")
            return False
            
        # Check if container exists but stopped
        container_check = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=ollama", "--format", "{{.Names}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        container_exists = container_check.returncode == 0 and container_check.stdout.strip()
        
        if container_exists:
            # Start existing container
            status_placeholder.info("Starting existing Ollama container...")
            start_result = subprocess.run(
                ["docker", "start", "ollama"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            if start_result.returncode != 0:
                status_placeholder.error(f"Failed to start existing Ollama container: {start_result.stderr.decode()}")
                return False
        else:
            # Create and run new container
            status_placeholder.info("Creating new Ollama container...")
            
            # Create models volume directory if it doesn't exist
            os.makedirs(OLLAMA_MODELS_DIR, exist_ok=True)
            
            # Get absolute path to models directory
            abs_models_path = os.path.abspath(OLLAMA_MODELS_DIR)
            
            # Run the container
            run_result = subprocess.run([
                "docker", "run", "-d",
                "--name", "ollama",
                "-p", "11434:11434",
                "-v", f"{abs_models_path}:/root/.ollama/models",
                "ollama/ollama"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if run_result.returncode != 0:
                status_placeholder.error(f"Failed to run Ollama container: {run_result.stderr.decode()}")
                return False
        
        # Wait for Ollama to start
        for i in range(15):  # Try for 15 seconds
            status_placeholder.info(f"Waiting for Ollama container to be ready... ({i+1}/15)")
            time.sleep(1)
            if is_ollama_running():
                status_placeholder.success("Ollama container started successfully!")
                return True
                
        status_placeholder.warning("Ollama container might not be ready yet. Please try again in a moment.")
        return False
        
    except Exception as e:
        if position == "sidebar":
            st.sidebar.error(f"Error starting Ollama container: {str(e)}")
        else:
            st.error(f"Error starting Ollama container: {str(e)}")
        return False

def ensure_ollama_running(position: str = "sidebar") -> bool:
    """
    Check if Ollama is running and try to start it if not.
    
    Args:
        position (str): Where to show notifications ("sidebar" or "content")
    
    Returns:
        bool: True if Ollama is running, False otherwise
    """
    notification_func = st.sidebar if position == "sidebar" else st
    
    # First check if Ollama is already running
    if is_ollama_running():
        return True
        
    # Then check if Ollama is running in Docker
    if is_docker_ollama_running():
        return True
        
    # Ollama is not running, show options to start it
    notification_func.warning("Ollama is not running. Please start it to use Ollama models.")
    
    # Create columns for options
    col1, col2 = notification_func.columns(2)
    
    with col1:
        if st.button("Start Local Ollama"):
            return start_local_ollama(position)
            
    with col2:
        if st.button("Start Docker Ollama"):
            return start_docker_ollama(position)
            
    return False

def get_ollama_models() -> List[Dict[str, Any]]:
    """
    Get a list of all available Ollama models with additional information.
    
    Returns:
        list: List of model information dictionaries
    """
    # Default library of models that can be pulled
    library_models = [
        {"id": "llama3", "name": "Llama 3", "description": "Meta's Llama 3 model", "pulled": False, "tags": ["chat", "general"]},
        {"id": "llama3:8b", "name": "Llama 3 (8B)", "description": "Meta's Llama 3 8B model", "pulled": False, "tags": ["chat", "general"]},
        {"id": "llama3:70b", "name": "Llama 3 (70B)", "description": "Meta's Llama 3 70B model", "pulled": False, "tags": ["chat", "general"]},
        {"id": "llama3:1b", "name": "Llama 3 (1B)", "description": "Meta's Llama 3 1B model", "pulled": False, "tags": ["chat", "fast"]},
        {"id": "deepseek-coder", "name": "DeepSeek Coder", "description": "Code-specialized LLM", "pulled": False, "tags": ["code", "technical"]},
        {"id": "deepseek-coder:6.7b", "name": "DeepSeek Coder (6.7B)", "description": "More reasonable sized code model", "pulled": False, "tags": ["code", "technical"]},
        {"id": "deepseek-llm", "name": "DeepSeek LLM", "description": "General purpose LLM", "pulled": False, "tags": ["chat", "general"]},
        {"id": "phi3:mini", "name": "Phi-3 Mini", "description": "Microsoft Phi-3 model", "pulled": False, "tags": ["chat", "fast"]},
        {"id": "phi3:medium", "name": "Phi-3 Medium", "description": "Microsoft's larger Phi-3 model", "pulled": False, "tags": ["chat", "general"]},
        {"id": "gemma:2b", "name": "Gemma 2B", "description": "Google's lightweight Gemma model", "pulled": False, "tags": ["chat", "fast"]},
        {"id": "gemma:7b", "name": "Gemma 7B", "description": "Google's Gemma 7B model", "pulled": False, "tags": ["chat", "general"]},
        {"id": "mistral", "name": "Mistral 7B", "description": "Mistral AI's 7B model", "pulled": False, "tags": ["chat", "general"]},
        {"id": "mistral:instruct", "name": "Mistral 7B Instruct", "description": "Instruction-tuned Mistral model", "pulled": False, "tags": ["chat", "general"]},
        {"id": "mixtral:8x7b", "name": "Mixtral 8x7B", "description": "Mistral's MoE model", "pulled": False, "tags": ["chat", "general"]},
        {"id": "codellama", "name": "Code Llama", "description": "Meta's code-specialized model", "pulled": False, "tags": ["code", "technical"]},
        {"id": "codellama:7b", "name": "Code Llama 7B", "description": "Smaller Code Llama model", "pulled": False, "tags": ["code", "technical"]},
        {"id": "qwen:14b", "name": "Qwen 14B", "description": "Alibaba's Qwen model", "pulled": False, "tags": ["chat", "general"]},
        {"id": "qwen:72b", "name": "Qwen 72B", "description": "Alibaba's large Qwen model", "pulled": False, "tags": ["chat", "general"]},
        {"id": "neural-chat", "name": "Neural Chat", "description": "Optimized conversational model", "pulled": False, "tags": ["chat", "customer-service"]},
        {"id": "stablelm:zephyr", "name": "StableLM Zephyr", "description": "Stability AI's Zephyr model", "pulled": False, "tags": ["chat", "general"]}
    ]
    
    # Check if Ollama is running
    if not is_ollama_running():
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
                    # Try to determine appropriate tags
                    tags = []
                    
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

def pull_ollama_model(model_name: str, position: str = "content") -> bool:
    """
    Pull an Ollama model with progress tracking.
    
    Args:
        model_name (str): Name of the model to pull
        position (str): Where to show notifications ("sidebar" or "content")
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Make sure Ollama is running
    if not ensure_ollama_running(position):
        return False
    
    try:
        notification_func = st.sidebar if position == "sidebar" else st
        progress_placeholder = notification_func.empty()
        progress_placeholder.info(f"Starting download of {model_name}...")
        
        # Start the pull operation
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": model_name}
        )
        
        if response.status_code != 200:
            progress_placeholder.error(f"Failed to start model download: {response.text}")
            return False
        
        # Monitor progress
        start_time = time.time()
        last_update_time = start_time
        progress_bar = notification_func.progress(0.0)
        
        pull_completed = False
        while not pull_completed and (time.time() - start_time) < 1800:  # 30 minute timeout
            try:
                # Check if model exists in list of models
                check_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
                if check_response.status_code == 200:
                    models = check_response.json().get("models", [])
                    if any(model["name"] == model_name for model in models):
                        progress_bar.progress(1.0)
                        progress_placeholder.success(f"Model {model_name} downloaded successfully!")
                        pull_completed = True
                        break
                
                # Update progress message every 5 seconds
                current_time = time.time()
                if current_time - last_update_time >= 5:
                    elapsed_mins = (current_time - start_time) / 60
                    progress_placeholder.info(f"Downloading {model_name}... {elapsed_mins:.1f} minutes elapsed")
                    # Update progress bar (simulate progress based on time)
                    simulated_progress = min(0.95, (current_time - start_time) / 1200)  # Max 20 minutes for 95%
                    progress_bar.progress(simulated_progress)
                    last_update_time = current_time
                
                time.sleep(1)
                
            except Exception as e:
                print(f"Error checking model status: {str(e)}")
                time.sleep(5)
        
        if not pull_completed:
            progress_placeholder.error(f"Download timeout for {model_name}. Model may still be downloading.")
            return False
        
        return True
        
    except Exception as e:
        notification_func = st.sidebar if position == "sidebar" else st
        notification_func.error(f"Error pulling model: {str(e)}")
        return False

def get_ollama_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific Ollama model.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        dict: Model information or None if not found
    """
    if not is_ollama_running():
        return None
        
    try:
        # Check if model exists
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code != 200:
            return None
            
        models = response.json().get("models", [])
        for model in models:
            if model["name"] == model_name:
                # Get model information
                info_response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/show",
                    json={"name": model_name}
                )
                
                if info_response.status_code == 200:
                    model_info = info_response.json()
                    return {
                        "id": model_name,
                        "name": model_name,
                        "description": model_info.get("description", ""),
                        "pulled": True,
                        "size": model.get("size", "Unknown"),
                        "family": model_info.get("family", ""),
                        "parameters": model_info.get("parameters", ""),
                        "template": model_info.get("template", ""),
                        "license": model_info.get("license", ""),
                        "details": model_info
                    }
                return {
                    "id": model_name,
                    "name": model_name,
                    "pulled": True,
                    "size": model.get("size", "Unknown")
                }
        
        # Model not found
        return None
        
    except Exception as e:
        print(f"Error getting model info: {str(e)}")
        return None

def initialize_ollama_model(model_name: str, temperature: float = 0.7, max_tokens: int = 512) -> Optional[Any]:
    """
    Initialize an Ollama model for use with LangChain.
    
    Args:
        model_name (str): Name of the model
        temperature (float): Temperature parameter
        max_tokens (int): Maximum tokens to generate
        
    Returns:
        LangchainLLM: The initialized model or None if initialization fails
    """
    try:
        from langchain_community.llms import Ollama
        
        # Make sure Ollama is running
        if not is_ollama_running():
            st.error("Ollama is not running. Please start it first.")
            return None
            
        # Check if model is pulled
        model_info = get_ollama_model_info(model_name)
        if not model_info:
            st.warning(f"Model {model_name} not found. You may need to pull it first.")
            return None
            
        # Initialize the model
        ollama_model = Ollama(
            model=model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
            num_predict=max_tokens
        )
        
        return ollama_model
        
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None