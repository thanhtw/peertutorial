import json
import requests
import os
from typing import Dict, List, Any, Optional

# Ollama API configuration
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "codellama")

def call_ollama(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Call the Ollama API with the given prompt.
    
    Args:
        prompt: The prompt to send to the model
        system_prompt: Optional system prompt for context
        
    Returns:
        The model's response as a string
    """
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
        return "Error communicating with the AI model. Please check if Ollama is running."

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
    Format your response in Markdown.
    """
    
    prompt = "Provide a comprehensive guide on how to conduct effective peer code reviews for students learning programming."
    
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
    You are an expert code reviewer providing feedback on a Java code snippet.
    Consider compilation results, style issues, and your programming expertise to provide a comprehensive review.
    Format your review in Markdown, highlighting both strengths and weaknesses of the code.
    Provide specific suggestions for improvements.
    """
    
    # Create a prompt that includes the code and analysis results
    prompt = f"""
    Please review the following Java code snippet:
    
    ```java
    {code_snippet}
    ```
    
    Compilation results:
    {"Successful" if compile_results["success"] else f"Failed with errors: {compile_results['errors']}"}
    
    Checkstyle results:
    {json.dumps(checkstyle_results["issues"], indent=2) if checkstyle_results["issues"] else "No style issues found."}
    
    Provide a comprehensive code review that addresses:
    1. Code correctness and potential bugs
    2. Code style and readability
    3. Design and architecture
    4. Performance considerations
    5. Specific recommendations for improvement
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
                "feedback": "The AI was unable to generate a structured analysis. Please try again.",
                "strengths": [],
                "improvements": [],
                "missed_issues": []
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