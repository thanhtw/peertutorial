"""
Module for compiling Java code.
This module provides functions to compile Java code and verify Java installation.
"""

import subprocess
import os
import tempfile
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


def compile_java(java_file_path: str) -> Dict[str, Any]:
    """
    Compile a Java file and return the compilation results.
    
    Args:
        java_file_path: Path to the Java file to compile
        
    Returns:
        A dictionary with compilation results:
        {
            "success": True/False,
            "errors": "Compilation error messages if any",
            "output": "Compilation output if any",
            "command": "Command that was executed"
        }
    """
    result = {
        "success": False,
        "errors": "",
        "output": "",
        "command": ""
    }
    
    try:
        # Ensure file exists
        if not os.path.exists(java_file_path):
            result["errors"] = f"File not found: {java_file_path}"
            return result
        
        # Extract class name from file for validation
        class_name = extract_class_name(java_file_path)
        
        # Get the directory for output
        directory = os.path.dirname(java_file_path)
        output_dir = os.path.join(directory, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Build the javac command
        command = ["javac", "-d", output_dir, java_file_path]
        result["command"] = " ".join(command)
        
        # Run compilation
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        result["output"] = process.stdout
        
        if process.returncode == 0:
            result["success"] = True
            
            # Verify the class file was created if we have the class name
            if class_name:
                class_file = os.path.join(output_dir, f"{class_name}.class")
                if not os.path.exists(class_file):
                    result["success"] = False
                    result["errors"] = f"Compilation appeared to succeed but {class_name}.class was not created."
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


def extract_class_name(file_path: str) -> Optional[str]:
    """
    Extract the class name from a Java file.
    
    Args:
        file_path: Path to the Java file
        
    Returns:
        The class name if found, None otherwise
    """
    try:
        class_name = None
        
        # Extract the filename without extension as a fallback
        base_name = os.path.basename(file_path)
        if base_name.endswith(".java"):
            fallback_name = base_name[:-5]
        else:
            fallback_name = None
        
        # Try to parse the file to extract the actual class name
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Pattern for public class name
            import re
            public_match = re.search(r'public\s+class\s+(\w+)', content)
            if public_match:
                class_name = public_match.group(1)
            else:
                # Try without public
                class_match = re.search(r'class\s+(\w+)', content)
                if class_match:
                    class_name = class_match.group(1)
                else:
                    class_name = fallback_name
        
        return class_name
    
    except Exception as e:
        print(f"Error extracting class name: {str(e)}")
        return None


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
        
        if process.returncode == 0:
            # Parse Java version
            version_output = process.stderr if process.stderr else process.stdout
            print(f"Java version: {version_output.splitlines()[0]}")
            return True
        
        return False
    except FileNotFoundError:
        return False


def get_java_version() -> Optional[str]:
    """
    Get the installed Java version.
    
    Returns:
        The Java version string if installed, None otherwise
    """
    try:
        process = subprocess.run(
            ["java", "-version"], 
            capture_output=True,
            text=True
        )
        
        if process.returncode == 0:
            version_output = process.stderr if process.stderr else process.stdout
            # Extract version info from first line
            return version_output.splitlines()[0] if version_output else None
        
        return None
    except FileNotFoundError:
        return None


def test_java_compiler(test_code: Optional[str] = None) -> Tuple[bool, str]:
    """
    Test if the Java compiler is working correctly.
    
    Args:
        test_code: Optional Java code to compile
        
    Returns:
        Tuple of (success, message)
    """
    if not ensure_java_installed():
        return False, "Java is not installed or not in path"
    
    # Create a simple test program if none provided
    if not test_code:
        test_code = """
        public class TestCompile {
            public static void main(String[] args) {
                System.out.println("Compilation test successful");
            }
        }
        """
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".java", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(test_code.encode())
        
        # Compile the test file
        result = compile_java(tmp_path)
        
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        if result["success"]:
            return True, "Java compiler is working correctly"
        else:
            return False, f"Java compiler test failed: {result['errors']}"
    
    except Exception as e:
        return False, f"Error testing Java compiler: {str(e)}"