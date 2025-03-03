"""
Ollama model utility module to check and initialize models.

This module provides functions to check available models in Ollama and 
initialize them for use in the application.
"""

import os
import requests
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple

# Constants
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODELS = ["llama3:1b", "llama3:8b", "phi3:mini", "gemma:2b", "deepseek-coder:6.7b"]


def check_ollama_server() -> bool:
    """
    Check if Ollama server is running.
    
    Returns:
        bool: True if server is running, False otherwise
    """
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def get_pulled_models() -> List[Dict[str, Any]]:
    """
    Get a list of all pulled models from Ollama.
    
    Returns:
        List[Dict[str, Any]]: List of model information dictionaries
    """
    if not check_ollama_server():
        return []
        
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            return response.json().get("models", [])
        return []
    except Exception as e:
        print(f"Error getting pulled models: {e}")
        return []


def initialize_best_model() -> Optional[str]:
    """
    Initialize the best available model from Ollama.
    
    Returns:
        Optional[str]: Name of initialized model or None if failed
    """
    # Get all pulled models
    pulled_models = get_pulled_models()
    if not pulled_models:
        return None
        
    # Extract model names
    model_names = [model["name"] for model in pulled_models]
    
    # Try to find the best model from our preferred list
    for model_name in DEFAULT_MODELS:
        if model_name in model_names:
            # Initialize this model
            return model_name
    
    # If none of our preferred models are available, use the first available model
    return model_names[0]


def check_and_pull_model(model_name: str) -> bool:
    """
    Check if a model is pulled and pull it if not.
    
    Args:
        model_name (str): Name of the model to check and pull
        
    Returns:
        bool: True if model is available (already pulled or successfully pulled), 
              False otherwise
    """
    # Get pulled models
    pulled_models = get_pulled_models()
    model_names = [model["name"] for model in pulled_models]
    
    # Check if model is already pulled
    if model_name in model_names:
        return True
        
    # Model is not pulled, try to pull it
    try:
        st.info(f"Pulling model {model_name}...")
        
        # Call Ollama API to pull the model
        response = requests.post(
            f"{OLLAMA_URL}/api/pull",
            json={"name": model_name}
        )
        
        if response.status_code == 200:
            st.success(f"Started pulling {model_name}")
            return True
        else:
            st.error(f"Failed to pull {model_name}")
            return False
    except Exception as e:
        st.error(f"Error pulling model: {str(e)}")
        return False


def get_model_details(model_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a model.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        Dict[str, Any]: Model details or empty dict if not found
    """
    if not check_ollama_server():
        return {}
        
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/show",
            json={"name": model_name}
        )
        
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        print(f"Error getting model details: {e}")
        return {}


def find_suitable_model() -> Tuple[str, List[Dict[str, Any]]]:
    """
    Find the most suitable model from pulled models.
    
    Returns:
        Tuple[str, List[Dict[str, Any]]]: Selected model name and list of all pulled models
    """
    pulled_models = get_pulled_models()
    
    if not pulled_models:
        return "", []
        
    # Extract model names
    model_names = [model["name"] for model in pulled_models]
    
    # Try to find the default model
    if "llama3:1b" in model_names:
        return "llama3:1b", pulled_models
        
    # Try to find other good small models
    for model_name in ["phi3:mini", "gemma:2b"]:
        if model_name in model_names:
            return model_name, pulled_models
            
    # Try medium-sized models    
    for model_name in ["llama3:8b", "deepseek-coder:6.7b"]:
        if model_name in model_names:
            return model_name, pulled_models
            
    # If no preferred models, return the first one
    return model_names[0], pulled_models