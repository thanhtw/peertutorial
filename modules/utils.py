import os
import subprocess
import tempfile
import json
from typing import Dict, List, Any, Optional

def setup_environment() -> bool:
    """
    Set up the required environment for the application.
    
    Returns:
        True if setup was successful, False otherwise
    """
    # Check if Java is installed
    java_installed = check_command("java -version")
    if not java_installed:
        print("Java is not installed or not in PATH.")
        return False
    
    # Check if Checkstyle is available
    checkstyle_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "checkstyle.jar")
    if not os.path.exists(checkstyle_path):
        print(f"Checkstyle JAR not found at {checkstyle_path}. Attempting to download...")
        from modules.checkstyle import download_checkstyle
        if not download_checkstyle():
            print("Failed to download Checkstyle.")
            return False
    
    # Check if Ollama is available
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    try:
        import requests
        response = requests.get(f"{ollama_url}/api/tags")
        if response.status_code != 200:
            print(f"Failed to connect to Ollama at {ollama_url}.")
            return False
        
        # Check if the required model is available
        ollama_model = os.environ.get("OLLAMA_MODEL", "codellama")
        models = response.json().get("models", [])
        if not any(model.get("name") == ollama_model for model in models):
            print(f"Model {ollama_model} not found in Ollama. Attempting to pull...")
            pull_response = requests.post(
                f"{ollama_url}/api/pull",
                json={"name": ollama_model}
            )
            if pull_response.status_code != 200:
                print(f"Failed to pull model {ollama_model}.")
                return False
    except Exception as e:
        print(f"Error checking Ollama: {str(e)}")
        return False
    
    return True

def check_command(command: str) -> bool:
    """
    Check if a command is available.
    
    Args:
        command: Command to check
        
    Returns:
        True if command is available, False otherwise
    """
    try:
        subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception:
        return False

def save_temp_file(content: str, extension: str = ".java") -> Optional[str]:
    """
    Save content to a temporary file.
    
    Args:
        content: Content to save
        extension: File extension
        
    Returns:
        Path to the temporary file, or None if an error occurred
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp_file:
            tmp_file.write(content.encode())
            return tmp_file.name
    except Exception as e:
        print(f"Error saving temporary file: {str(e)}")
        return None

def get_example_snippets() -> List[Dict[str, Any]]:
    """
    Get a list of example code snippets.
    
    Returns:
        List of dictionaries with snippet information
    """
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "code_examples")
    examples = []
    
    try:
        if os.path.exists(examples_dir):
            for filename in os.listdir(examples_dir):
                if filename.endswith(".java"):
                    file_path = os.path.join(examples_dir, filename)
                    with open(file_path, "r") as f:
                        content = f.read()
                    
                    examples.append({
                        "name": filename,
                        "path": file_path,
                        "content": content
                    })
    except Exception as e:
        print(f"Error reading example snippets: {str(e)}")
    
    return examples

def extract_class_name(java_code: str) -> Optional[str]:
    """
    Extract the class name from Java code.
    
    Args:
        java_code: Java code
        
    Returns:
        Class name if found, None otherwise
    """
    import re
    match = re.search(r"public\s+class\s+(\w+)", java_code)
    if match:
        return match.group(1)
    return None