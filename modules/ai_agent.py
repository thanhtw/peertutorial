"""
AI agent module for handling code review and analysis tasks.
This module interacts with Ollama to generate code reviews and feedback.
"""

import json
import requests
import os
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import time
import backoff
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ollama API configuration
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:1b")


def is_ollama_running() -> bool:
    """
    Check if Ollama is running at the configured URL.
    
    Returns:
        bool: True if Ollama is running, False otherwise
    """
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def start_ollama() -> bool:
    """
    Attempt to start the Ollama service.
    
    Returns:
        bool: True if successfully started, False otherwise
    """
    import platform
    import subprocess
    
    try:
        system = platform.system()
        
        if system == "Windows":
            # For Windows, try running the Ollama executable
            subprocess.Popen(
                ["ollama", "serve"],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        elif system == "Linux":
            # Try systemd service first
            try:
                subprocess.run(
                    ["systemctl", "--user", "start", "ollama"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5
                )
            except (subprocess.SubprocessError, FileNotFoundError):
                # Fallback to direct start
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
        elif system == "Darwin":  # macOS
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        else:
            print(f"Unsupported platform: {system}")
            return False
            
        # Wait for Ollama to start
        for i in range(10):
            time.sleep(1)
            if is_ollama_running():
                print("Ollama started successfully!")
                return True
                
        print("Ollama did not start within expected time")
        return False
        
    except Exception as e:
        print(f"Error starting Ollama: {str(e)}")
        return False


@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
def call_ollama(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Call the Ollama API with the given prompt, with retry capability.
    
    Args:
        prompt: The prompt to send to the model
        system_prompt: Optional system prompt for context
        
    Returns:
        The model's response as a string
    """
    # First check if Ollama is running
    if not is_ollama_running():
        if start_ollama():
            print("Ollama started successfully")
        else:
            error_msg = (
                "Ollama is not running or not accessible. Please:\n"
                "1. Ensure Ollama is installed\n"
                "2. Start Ollama using 'ollama serve' in a terminal\n"
                "3. Verify the connection URL is correct (currently: {OLLAMA_URL})\n"
                "4. Refresh this page and try again"
            )
            print(error_msg)
            return error_msg
    
    # Check if the model exists
    try:
        model_check = requests.get(f"{OLLAMA_URL}/api/tags")
        if model_check.status_code == 200:
            models = model_check.json().get("models", [])
            model_exists = any(model["name"] == OLLAMA_MODEL for model in models)
            if not model_exists:
                pull_msg = (
                    f"Model '{OLLAMA_MODEL}' is not available. Attempting to pull it...\n"
                )
                print(pull_msg)
                
                # Try to pull the model
                pull_response = requests.post(
                    f"{OLLAMA_URL}/api/pull",
                    json={"name": OLLAMA_MODEL}
                )
                
                if pull_response.status_code != 200:
                    return f"Failed to pull model '{OLLAMA_MODEL}'. Please pull it manually with 'ollama pull {OLLAMA_MODEL}'."
                
                # Wait for model to be pulled (up to 30 seconds)
                for i in range(30):
                    time.sleep(1)
                    check = requests.get(f"{OLLAMA_URL}/api/tags")
                    if check.status_code == 200:
                        models = check.json().get("models", [])
                        if any(model["name"] == OLLAMA_MODEL for model in models):
                            print(f"Model '{OLLAMA_MODEL}' successfully pulled!")
                            break
                else:
                    return f"Timeout waiting for model '{OLLAMA_MODEL}' to be pulled. Please try again later."
    except Exception as e:
        print(f"Error checking available models: {e}")
    
    # Call the Ollama API
    headers = {"Content-Type": "application/json"}
    data = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    
    if system_prompt:
        data["system"] = system_prompt
    
    try:
        response = requests.post(f"{OLLAMA_URL}/api/generate", headers=headers, json=data)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return (
            "Error communicating with the Ollama API.\n\n"
            f"Details: {str(e)}\n\n"
            "Please check if:\n"
            f"1. Ollama is running correctly\n"
            f"2. The model '{OLLAMA_MODEL}' is properly installed\n"
            f"3. The URL '{OLLAMA_URL}' is correct"
        )


def get_code_review_knowledge() -> str:
    """
    Generate knowledge about peer code review.
    
    Returns:
        A formatted string with code review knowledge
    """
    system_prompt = """
    You are an expert in software engineering and code review. 
    Provide comprehensive knowledge about effective peer code reviews including best practices, 
    common pitfalls, and how to give constructive feedback.
    Focus specifically on Java code reviews, covering style conventions, common pitfalls, and design patterns.
    Format your response in Markdown.
    """
    
    prompt = "Provide a comprehensive guide on how to conduct effective peer code reviews for students learning Java programming."
    
    return call_ollama(prompt, system_prompt)


def generate_code_snippet() -> str:
    """
    Generate a Java code snippet with intentional issues for review practice.
    
    Returns:
        A Java code snippet as a string
    """
    system_prompt = """
    You are a programming instructor creating educational materials.
    Generate a Java code snippet (a complete class) that has some intentional issues for students to find during code review.
    Include a mix of issues such as:
    1. Potential bugs or logic errors
    2. Style violations (naming conventions, formatting)
    3. Performance issues
    4. Poor design choices
    
    The code should be compilable but have room for improvement.
    Make sure the class name matches the filename that would be required in Java.
    """
    
    prompt = "Generate a Java code snippet with intentional issues for a peer code review exercise. The code should be a complete class, between 30-50 lines long."
    
    return call_ollama(prompt, system_prompt)


def get_ai_review(
    code_snippet: str, 
    compile_results: Dict[str, Any], 
    checkstyle_results: Dict[str, Any]
) -> str:
    """
    Generate an AI review of the code snippet based on compilation and checkstyle results.
    
    Args:
        code_snippet: The Java code snippet to review
        compile_results: Results from compiling the code
        checkstyle_results: Results from running checkstyle
        
    Returns:
        A formatted review string
    """
    system_prompt = """
    You are an expert Java code reviewer providing feedback on a code snippet.
    Consider compilation results, style issues, and your programming expertise to provide a comprehensive review.
    Focus on Java best practices, object-oriented design principles, and coding standards.
    Format your review in Markdown, highlighting both strengths and weaknesses of the code.
    Provide specific suggestions for improvements with examples where appropriate.
    """
    
    # Create a prompt that includes the code and analysis results
    compilation_status = "Successful" if compile_results["success"] else f"Failed with errors: {compile_results['errors']}"
    checkstyle_issues = json.dumps(checkstyle_results["issues"], indent=2) if checkstyle_results["issues"] else "No style issues found."
    
    prompt = f"""
    Please review the following Java code snippet:
    
    ```java
    {code_snippet}
    ```
    
    Compilation results:
    {compilation_status}
    
    Checkstyle results:
    {checkstyle_issues}
    
    Provide a comprehensive code review that addresses:
    1. Code correctness and potential bugs
    2. Code style and readability
    3. Design and architecture
    4. Performance considerations
    5. Specific recommendations for improvement
    
    For each issue you identify, please explain:
    - What the issue is
    - Why it's a problem 
    - How to fix it with a code example
    """
    
    return call_ollama(prompt, system_prompt)


def compare_reviews(ai_review: str, student_review: str) -> Dict[str, Any]:
    """
    Compare the AI-generated review with the student's review.
    
    Args:
        ai_review: The AI-generated code review
        student_review: The student's code review
        
    Returns:
        A dictionary with comparison results
    """
    system_prompt = """
    You are an experienced programming instructor evaluating a student's code review skills.
    Compare the student's review with the AI-generated review to provide constructive feedback.
    Focus on what the student did well and what they missed or could improve.
    Be encouraging but thorough in your assessment.
    """
    
    prompt = f"""
    Compare the following student code review with the expert AI review.
    
    AI REVIEW:
    {ai_review}
    
    STUDENT REVIEW:
    {student_review}
    
    Analyze the comparison and provide:
    1. Overall feedback on the student's review quality
    2. A list of strengths in the student's review
    3. A list of areas for improvement
    4. A list of important issues that the student may have missed but were caught in the AI review
    
    Return your analysis in JSON format with the following structure:
    {{
        "feedback": "Overall feedback...",
        "strengths": ["Strength 1", "Strength 2", ...],
        "improvements": ["Improvement 1", "Improvement 2", ...],
        "missed_issues": ["Missed issue 1", "Missed issue 2", ...]
    }}
    """
    
    response = call_ollama(prompt, system_prompt)
    
    # Try to parse the response as JSON
    try:
        # Find JSON content in the response (in case it includes explanatory text)
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)
            return result
        else:
            # Fallback if proper JSON is not found
            return {
                "feedback": response,
                "strengths": [],
                "improvements": ["Try to structure your review more clearly"],
                "missed_issues": ["Unable to analyze specific missed issues"]
            }
    except json.JSONDecodeError:
        # Handle the case where response is not valid JSON
        # Generate structured feedback manually
        return {
            "feedback": response,
            "strengths": ["Your review was received"],
            "improvements": ["Try to be more specific in your feedback"],
            "missed_issues": ["Unable to analyze missed issues"]
        }


def set_ollama_model(model_name: str) -> bool:
    """
    Set the Ollama model to use for code reviews.
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        True if successful, False otherwise
    """
    global OLLAMA_MODEL
    
    # Check if model exists
    try:
        model_check = requests.get(f"{OLLAMA_URL}/api/tags")
        if model_check.status_code == 200:
            models = model_check.json().get("models", [])
            model_exists = any(model["name"] == model_name for model in models)
            
            if model_exists:
                OLLAMA_MODEL = model_name
                return True
            else:
                # Try to pull the model
                print(f"Model '{model_name}' not found. Attempting to pull...")
                pull_response = requests.post(
                    f"{OLLAMA_URL}/api/pull",
                    json={"name": model_name}
                )
                
                if pull_response.status_code != 200:
                    print(f"Failed to pull model '{model_name}'")
                    return False
                
                # Wait for model to be pulled (up to 30 seconds)
                for i in range(30):
                    time.sleep(1)
                    check = requests.get(f"{OLLAMA_URL}/api/tags")
                    if check.status_code == 200:
                        models = check.json().get("models", [])
                        if any(model["name"] == model_name for model in models):
                            OLLAMA_MODEL = model_name
                            return True
                
                return False
    except Exception as e:
        print(f"Error setting Ollama model: {e}")
        return False


def get_available_models() -> List[Dict[str, Any]]:
    """
    Get a list of available Ollama models.
    
    Returns:
        List of model information dictionaries
    """
    # Default models that can be pulled
    default_models = [
        {"id": "llama3:1b", "name": "Llama 3 (1B)", "description": "Meta's Llama 3 1B model", "pulled": False, "recommended": True},
        {"id": "llama3:8b", "name": "Llama 3 (8B)", "description": "Meta's Llama 3 8B model", "pulled": False, "recommended": True},
        {"id": "deepseek-coder:6.7b", "name": "DeepSeek Coder (6.7B)", "description": "Code-specialized model", "pulled": False, "recommended": True},
        {"id": "phi3:mini", "name": "Phi-3 Mini", "description": "Microsoft Phi-3 model", "pulled": False, "recommended": True},
        {"id": "gemma:2b", "name": "Gemma 2B", "description": "Google's lightweight Gemma model", "pulled": False, "recommended": True}
    ]
    
    # Check if Ollama is running
    if not is_ollama_running():
        return default_models
    
    try:
        # Get list of already pulled models
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            pulled_models = response.json().get("models", [])
            pulled_ids = [model["name"] for model in pulled_models]
            
            # Mark models as pulled if they exist
            for model in default_models:
                if model["id"] in pulled_ids:
                    model["pulled"] = True
            
            # Add any pulled models that aren't in our default list
            for pulled_model in pulled_models:
                model_id = pulled_model["name"]
                if not any(model["id"] == model_id for model in default_models):
                    # Extract size information
                    size_str = pulled_model.get("size", "Unknown")
                    # Add to the list
                    default_models.append({
                        "id": model_id,
                        "name": model_id,
                        "description": f"Size: {size_str}",
                        "pulled": True,
                        "recommended": False
                    })
        
        return default_models
    except Exception as e:
        print(f"Error getting Ollama models: {str(e)}")
        return default_models


def pull_model(model_name: str) -> bool:
    """
    Pull an Ollama model.
    
    Args:
        model_name: Name of the model to pull
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not is_ollama_running():
        if not start_ollama():
            return False
    
    try:
        print(f"Pulling model {model_name}...")
        response = requests.post(
            f"{OLLAMA_URL}/api/pull",
            json={"name": model_name}
        )
        
        if response.status_code != 200:
            print(f"Failed to start model download: {response.text}")
            return False
        
        # Wait for model to be pulled (checking periodically)
        max_wait_time = 300  # 5 minute timeout
        start_time = time.time()
        
        while (time.time() - start_time) < max_wait_time:
            # Check if model exists in list of models
            check_response = requests.get(f"{OLLAMA_URL}/api/tags")
            if check_response.status_code == 200:
                models = check_response.json().get("models", [])
                if any(model["name"] == model_name for model in models):
                    print(f"Model {model_name} pulled successfully!")
                    return True
            
            # Wait before checking again
            time.sleep(5)
            
        print(f"Timeout pulling model {model_name}")
        return False
        
    except Exception as e:
        print(f"Error pulling model: {str(e)}")
        return False