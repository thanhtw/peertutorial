"""
Code Generator Agent for Peer Code Review Tutorial System

This agent is responsible for generating Java code snippets with intentional errors
for students to review, as well as expert reviews of that code.
"""

import json
import random
import requests
import os
from typing import Dict, List, Any, Optional
import backoff

class CodeGeneratorAgent:
    """
    Code Generator Agent that creates code snippets with intentional errors.
    
    This agent:
    1. Generates Java code with specific types of errors
    2. Creates expert reviews for comparison
    3. Varies the difficulty and focus areas of generated code
    """
    
    def __init__(self):
        """Initialize the Code Generator Agent."""
        self.model_name = None
        self.temperature = 0.7
        self.max_tokens = 1024
        self.ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    
    def set_model(self, model_name: str, temperature: float = 0.7, max_tokens: int = 1024) -> None:
        """
        Set the LLM model parameters.
        
        Args:
            model_name: Name of the model to use
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
    def _call_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call the Ollama API with the given prompt.
        
        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt for context
            
        Returns:
            The model's response as a string
        """
        if not self.model_name:
            return "Error: Model not initialized"
        
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": self.temperature,
            "num_predict": self.max_tokens
        }
        
        if system_prompt:
            data["system"] = system_prompt
        
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", headers=headers, json=data)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return f"Error communicating with the Ollama API: {str(e)}"
    
    def generate_code_snippet(
        self, 
        difficulty: str = "medium", 
        focus_areas: Optional[List[str]] = None,
        build_errors: Optional[Dict] = None,
        checkstyle_errors: Optional[Dict] = None
    ) -> str:
        """
        Generate a Java code snippet with intentional issues for review.
        
        Args:
            difficulty: Difficulty level (easy, medium, hard)
            focus_areas: List of areas to focus on (e.g., ["naming", "exceptions", "performance"])
            build_errors: Dictionary of common build errors to include
            checkstyle_errors: Dictionary of common checkstyle errors to include
            
        Returns:
            A Java code snippet as a string
        """
        # Default focus areas if none provided
        if not focus_areas:
            focus_areas = ["naming", "structure", "error_handling"]
        
        # Select specific errors to include based on difficulty and focus areas
        specific_errors = self._select_errors_for_snippet(difficulty, focus_areas, build_errors, checkstyle_errors)
        
        # Create system prompt with detailed instructions
        system_prompt = self._create_code_generation_prompt(difficulty, focus_areas, specific_errors)
        
        # User prompt for code generation
        user_prompt = f"""
        Generate a Java code snippet for a peer code review exercise with the following characteristics:
        - Difficulty level: {difficulty}
        - Focus areas: {', '.join(focus_areas)}
        - Length: 30-50 lines of code
        - Include a mix of {len(specific_errors)} issues as specified in the system prompt
        - The code should be complete and compilable (except for intentional build errors)
        - Use meaningful variable names and realistic code structure
        """
        
        # Generate code snippet
        code_snippet = self._call_ollama(user_prompt, system_prompt)
        
        return code_snippet
    
    def _select_errors_for_snippet(
        self, 
        difficulty: str, 
        focus_areas: List[str],
        build_errors: Optional[Dict] = None,
        checkstyle_errors: Optional[Dict] = None
    ) -> List[Dict[str, str]]:
        """
        Select specific errors to include in the code snippet.
        
        Args:
            difficulty: Difficulty level
            focus_areas: Areas to focus on
            build_errors: Dictionary of build errors
            checkstyle_errors: Dictionary of checkstyle errors
            
        Returns:
            List of selected errors to include
        """
        selected_errors = []
        
        # Determine number of errors based on difficulty
        if difficulty == "easy":
            num_errors = random.randint(2, 3)
        elif difficulty == "medium":
            num_errors = random.randint(3, 5)
        else:  # hard
            num_errors = random.randint(5, 7)
        
        # If error databases are provided, use them to select specific errors
        if build_errors and checkstyle_errors:
            # Map focus areas to error categories
            error_categories = {
                "naming": ["NamingConventionChecks"],
                "structure": ["BlockChecks", "WhitespaceAndFormattingChecks"],
                "error_handling": ["CompileTimeErrors", "RuntimeErrors"],
                "imports": ["ImportChecks"],
                "javadoc": ["JavadocChecks"],
                "metrics": ["MetricsChecks"],
                "code_quality": ["CodeQualityChecks", "LogicalErrors"],
                "performance": ["LogicalErrors"],
                "security": ["WarningsAndHints", "LogicalErrors"]
            }
            
            # Select errors from appropriate categories based on focus areas
            build_categories = ["CompileTimeErrors", "RuntimeErrors", "LogicalErrors", "WarningsAndHints"]
            checkstyle_categories = [
                "NamingConventionChecks", "WhitespaceAndFormattingChecks", "BlockChecks", 
                "ImportChecks", "JavadocChecks", "MetricsChecks", "CodeQualityChecks", 
                "FileStructureChecks", "MiscellaneousChecks"
            ]
            
            # Determine how many of each type to include
            num_build_errors = num_errors // 2
            num_checkstyle_errors = num_errors - num_build_errors
            
            # Prioritize categories from focus areas
            relevant_build_categories = []
            relevant_checkstyle_categories = []
            
            for focus in focus_areas:
                if focus in error_categories:
                    for category in error_categories[focus]:
                        if category in build_categories:
                            relevant_build_categories.append(category)
                        elif category in checkstyle_categories:
                            relevant_checkstyle_categories.append(category)
            
            # Fallback to all categories if none match focus areas
            if not relevant_build_categories:
                relevant_build_categories = build_categories
            if not relevant_checkstyle_categories:
                relevant_checkstyle_categories = checkstyle_categories
            
            # Select build errors
            for _ in range(num_build_errors):
                category = random.choice(relevant_build_categories)
                if category in build_errors:
                    error = random.choice(build_errors[category])
                    selected_errors.append({
                        "type": "build",
                        "category": category,
                        "name": error["error_name"],
                        "description": error["description"]
                    })
            
            # Select checkstyle errors
            for _ in range(num_checkstyle_errors):
                category = random.choice(relevant_checkstyle_categories)
                if category in checkstyle_errors:
                    error = random.choice(checkstyle_errors[category])
                    selected_errors.append({
                        "type": "checkstyle",
                        "category": category,
                        "name": error["check_name"],
                        "description": error["description"]
                    })
        else:
            # Fallback if no error databases provided
            generic_errors = [
                {"type": "naming", "description": "Use incorrect naming conventions"},
                {"type": "structure", "description": "Use poor code structure or formatting"},
                {"type": "error_handling", "description": "Add inadequate error handling"},
                {"type": "performance", "description": "Include performance issues"},
                {"type": "logic", "description": "Add logical errors or bugs"}
            ]
            selected_errors = random.sample(generic_errors, min(num_errors, len(generic_errors)))
        
        return selected_errors
    
    def _create_code_generation_prompt(
        self, 
        difficulty: str, 
        focus_areas: List[str],
        specific_errors: List[Dict[str, str]]
    ) -> str:
        """
        Create a detailed system prompt for code generation.
        
        Args:
            difficulty: Difficulty level
            focus_areas: Focus areas
            specific_errors: List of specific errors to include
            
        Returns:
            System prompt as a string
        """
        system_prompt = f"""
        You are a programming instructor creating educational code review exercises.
        Generate a Java code snippet (a complete class) that has intentional issues for students to find.
        
        Difficulty level: {difficulty}
        Focus areas: {', '.join(focus_areas)}
        
        Include the following specific issues in your code:
        {json.dumps(specific_errors, indent=2)}
        
        Guidelines:
        1. Make sure the class name follows Java conventions and would match an expected filename
        2. The code should look realistic and be meaningful, not just a random collection of statements
        3. Include proper class structure with fields, methods, and appropriate access modifiers
        4. Ensure issues are subtle and educational, not just obvious errors
        5. For "easy" difficulty, make issues more obvious
        6. For "medium" difficulty, make issues somewhat subtle
        7. For "hard" difficulty, make issues quite subtle and require careful review
        8. The code should be between 30-50 lines long
        9. Include comments where appropriate, but don't call out the errors in comments
        
        Return only the Java code without any additional explanation.
        """
        return system_prompt
    
    def generate_expert_review(
        self, 
        code_snippet: str, 
        compile_results: Dict[str, Any], 
        checkstyle_results: Dict[str, Any],
        problem_analysis: Dict[str, Any]
    ) -> str:
        """
        Generate an expert review of the code snippet.
        
        Args:
            code_snippet: The Java code snippet
            compile_results: Results from compiling the code
            checkstyle_results: Results from running checkstyle
            problem_analysis: Analysis of the problems in the code
            
        Returns:
            A formatted review string
        """
        system_prompt = """
        You are an expert Java code reviewer providing feedback on a code snippet.
        Your task is to write a detailed, educational review that identifies all issues in the code.
        Your review should be thorough, clear, and helpful for learning purposes.
        Format your review in Markdown, with clear sections and bullet points.
        For each issue, explain:
        1. What the issue is
        2. Why it's a problem 
        3. How to fix it with a specific code example
        
        Include both positive aspects of the code and areas for improvement.
        Be specific and provide actual code examples for fixes.
        """
        
        # Compile status summary for the prompt
        compilation_status = "successful" if compile_results["success"] else f"failed with errors: {compile_results['errors']}"
        
        # Checkstyle issues summary
        if checkstyle_results["issues"]:
            checkstyle_summary = f"{len(checkstyle_results['issues'])} style issues found"
        else:
            checkstyle_summary = "no style issues found"
        
        # Create a detailed prompt that includes the code and analysis results
        prompt = f"""
        Please review the following Java code snippet:
        
        ```java
        {code_snippet}
        ```
        
        Compilation: {compilation_status}
        
        Checkstyle: {checkstyle_summary}
        
        Known issues in the code (reference for your review):
        {json.dumps(problem_analysis["issues"], indent=2)}
        
        Write a comprehensive expert review that addresses:
        1. Code correctness and potential bugs
        2. Code style and readability
        3. Design and architecture
        4. Performance considerations
        5. Security implications (if relevant)
        
        Provide specific recommendations with code examples for each issue.
        """
        
        # Generate expert review
        expert_review = self._call_ollama(prompt, system_prompt)
        
        return expert_review