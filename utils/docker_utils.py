"""
Docker utilities for managing Ollama containers.

This module provides functions to check, start, and interact with Docker containers,
specifically focused on managing Ollama containers.
"""

import os
import subprocess
import time
import streamlit as st
from typing import Optional, Dict, Any, List

# Docker image for Ollama
OLLAMA_DOCKER_IMAGE = "ollama/ollama:latest"

def is_docker_available() -> bool:
    """
    Check if Docker is installed and available.
    
    Returns:
        bool: True if Docker is installed, False otherwise
    """
    try:
        result = subprocess.run(
            ["docker", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False

def check_ollama_running() -> bool:
    """
    Check if an Ollama container is running.
    
    Returns:
        bool: True if Ollama container is running, False otherwise
    """
    if not is_docker_available():
        return False
        
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=ollama", "--format", "{{.Names}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True
        )
        
        return result.returncode == 0 and "ollama" in result.stdout
    except Exception:
        return False

def get_ollama_container_id() -> Optional[str]:
    """
    Get the ID of the Ollama container if it exists.
    
    Returns:
        Optional[str]: Container ID or None if not found
    """
    if not is_docker_available():
        return None
        
    try:
        # Check for running container first
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=ollama", "--format", "{{.ID}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        
        # Check for stopped container
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=ollama", "--format", "{{.ID}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
            
        return None
    except Exception:
        return None

def start_ollama_container() -> bool:
    """
    Start an existing but stopped Ollama container.
    
    Returns:
        bool: True if successful, False otherwise
    """
    container_id = get_ollama_container_id()
    if not container_id:
        return False
        
    try:
        result = subprocess.run(
            ["docker", "start", container_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        
        return result.returncode == 0
    except Exception:
        return False

def run_ollama_container(position_noti: str = "sidebar") -> bool:
    """
    Create and run an Ollama container.
    
    Args:
        position_noti (str): Where to show notifications ("sidebar" or "content")
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not is_docker_available():
        notification = st.sidebar if position_noti == "sidebar" else st
        notification.error("Docker is not installed or not available.")
        return False
    
    # Check if container already exists
    container_id = get_ollama_container_id()
    
    if container_id:
        # Container exists, try to start it
        notification = st.sidebar if position_noti == "sidebar" else st
        notification.info(f"Ollama container found (ID: {container_id}). Starting...")
        return start_ollama_container()
    
    # Create models directory
    models_dir = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
    ollama_models_dir = os.path.join(models_dir, "ollama")
    os.makedirs(ollama_models_dir, exist_ok=True)
    
    # Get absolute path
    abs_models_path = os.path.abspath(ollama_models_dir)
    
    # Run the container
    try:
        notification = st.sidebar if position_noti == "sidebar" else st
        notification.info(f"Creating new Ollama container with volume: {abs_models_path}")
        
        result = subprocess.run(
            [
                "docker", "run", 
                "-d",
                "--name", "ollama",
                "-p", "11434:11434",
                "-v", f"{abs_models_path}:/root/.ollama",
                OLLAMA_DOCKER_IMAGE
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True
        )
        
        if result.returncode != 0:
            notification.error(f"Failed to create Ollama container: {result.stderr}")
            return False
            
        notification.success(f"Ollama container created (ID: {result.stdout.strip()})")
        
        # Wait for container to be ready
        time.sleep(2)
        return True
        
    except Exception as e:
        notification = st.sidebar if position_noti == "sidebar" else st
        notification.error(f"Error creating Ollama container: {str(e)}")
        return False

def get_ollama_container_status() -> Dict[str, Any]:
    """
    Get detailed status of the Ollama container.
    
    Returns:
        Dict[str, Any]: Status information
    """
    status = {
        "running": False,
        "container_id": None,
        "ports": None,
        "image": None,
        "created": None,
        "status": "Not found"
    }
    
    if not is_docker_available():
        return status
        
    container_id = get_ollama_container_id()
    if not container_id:
        return status
        
    status["container_id"] = container_id
    
    try:
        # Get container info
        result = subprocess.run(
            ["docker", "inspect", container_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True
        )
        
        if result.returncode == 0:
            import json
            container_info = json.loads(result.stdout)
            
            if container_info and len(container_info) > 0:
                info = container_info[0]
                status["running"] = info.get("State", {}).get("Running", False)
                status["image"] = info.get("Config", {}).get("Image", "unknown")
                status["created"] = info.get("Created", "unknown")
                status["status"] = info.get("State", {}).get("Status", "unknown")
                
                # Get port mappings
                ports = info.get("NetworkSettings", {}).get("Ports", {})
                if ports:
                    status["ports"] = ports
        
        return status
    except Exception:
        return status

def pull_ollama_model(model_name: str, position_noti: str = "sidebar") -> bool:
    """
    Pull an Ollama model using Docker.
    
    Args:
        model_name (str): Name of the model to pull
        position_noti (str): Where to show notifications ("sidebar" or "content")
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not check_ollama_running():
        notification = st.sidebar if position_noti == "sidebar" else st
        notification.error("Ollama container is not running. Please start it first.")
        return False
        
    try:
        notification = st.sidebar if position_noti == "sidebar" else st
        progress_bar = notification.progress(0.0)
        status_text = notification.empty()
        status_text.info(f"Pulling model: {model_name}...")
        
        # Execute 'ollama pull' command inside the container
        result = subprocess.run(
            ["docker", "exec", "ollama", "ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True
        )
        
        if result.returncode != 0:
            status_text.error(f"Failed to pull model: {result.stderr}")
            return False
            
        progress_bar.progress(1.0)
        status_text.success(f"Successfully pulled model: {model_name}")
        return True
        
    except Exception as e:
        notification = st.sidebar if position_noti == "sidebar" else st
        notification.error(f"Error pulling model: {str(e)}")
        return False

def list_ollama_models() -> List[Dict[str, Any]]:
    """
    List all available Ollama models in the container.
    
    Returns:
        List[Dict[str, Any]]: List of model information
    """
    if not check_ollama_running():
        return []
        
    try:
        result = subprocess.run(
            ["docker", "exec", "ollama", "ollama", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True
        )
        
        if result.returncode != 0:
            return []
            
        models = []
        lines = result.stdout.strip().split("\n")
        
        # Skip header row
        if len(lines) > 1:
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 3:
                    models.append({
                        "name": parts[0],
                        "id": parts[0],
                        "size": parts[1],
                        "pulled": True,
                        "modified": " ".join(parts[2:])
                    })
                    
        return models
    except Exception:
        return []