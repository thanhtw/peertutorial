import subprocess
import os
import tempfile
from typing import Dict, Any

def compile_java(java_file_path: str) -> Dict[str, Any]:
    """
    Compile a Java file and return the compilation results.
    
    Args:
        java_file_path: Path to the Java file to compile
        
    Returns:
        A dictionary with compilation results:
        {
            "success": True/False,
            "errors": "Compilation error messages if any"
        }
    """
    result = {
        "success": False,
        "errors": ""
    }
    
    try:
        # Ensure we have file path and it exists
        if not os.path.exists(java_file_path):
            result["errors"] = f"File not found: {java_file_path}"
            return result
        
        # Get the directory and file name
        directory = os.path.dirname(java_file_path)
        
        # Run javac command
        process = subprocess.run(
            ["javac", java_file_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if process.returncode == 0:
            result["success"] = True
        else:
            result["success"] = False
            result["errors"] = process.stderr
        
        return result
    
    except subprocess.TimeoutExpired:
        result["errors"] = "Compilation timed out after 30 seconds"
        return result
    
    except FileNotFoundError:
        result["errors"] = "Java compiler (javac) not found. Please make sure Java is installed."
        return result
    
    except Exception as e:
        result["errors"] = f"An error occurred during compilation: {str(e)}"
        return result

def ensure_java_installed() -> bool:
    """
    Check if Java is installed and available.
    
    Returns:
        True if Java is installed, False otherwise
    """
    try:
        process = subprocess.run(
            ["java", "-version"], 
            capture_output=True,
            text=True
        )
        return process.returncode == 0
    except FileNotFoundError:
        return False

def install_java() -> bool:
    """
    Install Java if not already installed.
    This is a placeholder and requires system-specific implementation.
    
    Returns:
        True if installation was successful, False otherwise
    """
    print("Attempting to install Java...")
    
    try:
        # This is system-specific. The example below is for Ubuntu/Debian.
        # Replace with appropriate command for your system
        process = subprocess.run(
            ["apt-get", "update", "-y"],
            capture_output=True,
            text=True
        )
        
        if process.returncode != 0:
            print(f"Failed to update package lists: {process.stderr}")
            return False
        
        process = subprocess.run(
            ["apt-get", "install", "-y", "default-jdk"],
            capture_output=True,
            text=True
        )
        
        if process.returncode != 0:
            print(f"Failed to install Java: {process.stderr}")
            return False
        
        return ensure_java_installed()
    
    except Exception as e:
        print(f"An error occurred while installing Java: {str(e)}")
        return False

# If this file is run directly, check if Java is installed
if __name__ == "__main__":
    if ensure_java_installed():
        print("Java is installed and available.")
    else:
        print("Java is not installed. Attempting to install...")
        if install_java():
            print("Java was successfully installed.")
        else:
            print("Failed to install Java. Please install it manually.")