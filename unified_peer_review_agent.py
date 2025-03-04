"""
Unified AI Agent for Peer Code Review Learning System

This module provides a single AI agent that replaces the multi-agent architecture,
combining functionality from the Orchestrator, Code Generator, Problem Tracker,
Review Analytics, Feedback, and Progress Tracker agents into one unified class.
"""

import os
import json
import time
import tempfile
import hashlib
import re
import requests
import backoff
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import platform
import random

class UnifiedPeerReviewAgent:
    """
    Unified AI Agent for Peer Code Review Learning System
    
    This agent combines all the functionality previously spread across multiple agents:
    - Code generation with intentional errors
    - Problem tracking and analysis
    - Review comparison and analytics
    - Personalized feedback generation
    - Learning progress tracking
    - Orchestration of the overall workflow
    """
    
    def __init__(self):
        """Initialize the unified agent with all necessary components."""
        # Model and API settings
        self.model_name = None
        self.temperature = 0.7
        self.max_tokens = 1024
        self.ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        
        # System state
        self.model_initialized = False
        self.selected_model = os.environ.get("OLLAMA_MODEL", "llama3:1b")
        
        # Learning history for progress tracking
        self.learning_history = {
            "exercises": [],
            "performance": {},
            "strengths": [],
            "weaknesses": [],
            "completed_exercises": 0,
            "current_level": "beginner",
            "progress_by_category": defaultdict(list),
            "last_updated": time.time()
        }
        
        # Issue tracking
        self.issues_database = {}
        
        # Load error databases
        self._load_error_databases()
        
        # Templates for feedback
        self._load_feedback_templates()
    
    def _load_error_databases(self) -> bool:
        """Load build errors and checkstyle errors from JSON files."""
        try:
            # Get the base directory
            base_dir = Path(__file__).parent
            
            # Load build errors
            build_errors_path = base_dir / "build_errors.json"
            with open(build_errors_path, 'r') as f:
                self.build_errors = json.load(f)
            
            # Load checkstyle errors
            checkstyle_errors_path = base_dir / "checkstyle_error.json"
            with open(checkstyle_errors_path, 'r') as f:
                self.checkstyle_errors = json.load(f)
                
            return True
        except Exception as e:
            print(f"Error loading error databases: {str(e)}")
            self.build_errors = {}
            self.checkstyle_errors = {}
            return False
    
    def _load_feedback_templates(self) -> None:
        """Load and initialize feedback templates for different scenarios."""
        # Templates for different types of feedback
        self.templates = {
            "positive": [
                "Great job finding {issue_type}! Your attention to {focus} is excellent.",
                "You correctly identified the {issue_type}. This is important because {reason}.",
                "Well done spotting the {issue_type}. This shows good understanding of {concept}."
            ],
            "encouragement": [
                "You're making good progress. Keep focusing on {focus_area} in your next review.",
                "Your review shows improvement in {strength_area}. Next time, try to spot issues related to {focus_area}.",
                "You've mastered {mastered_area}! Now let's work on strengthening your {focus_area} skills."
            ],
            "missed_high": [
                "You missed a critical issue: {issue_description}. This is important because {reason}.",
                "An important issue you didn't catch was the {issue_type} on line {line}. This could cause {consequence}.",
                "Take a closer look at line {line}. There's a {issue_type} that needs attention because {reason}."
            ],
            "missed_medium": [
                "Consider checking for {issue_type} issues in your reviews. You missed one at line {line}.",
                "Another thing to look for: {issue_description} around line {line}.",
                "The code also has a {issue_type} issue that wasn't mentioned in your review."
            ],
            "explanation": [
                "When you see {pattern}, it often indicates {issue_type}. This is problematic because {reason}.",
                "A common pitfall in Java is {issue_description}. You can identify this by looking for {pattern}.",
                "The {issue_type} issue occurs when {explanation}. A best practice is to {solution}."
            ],
            "improvement": [
                "To improve your code reviews, try {technique}.",
                "Next time, consider using this approach: {approach}.",
                "A helpful strategy for finding {issue_type} issues is to {strategy}."
            ]
        }
    
    #
    # Model Initialization and API Methods
    #
    
    def initialize_model(self, model_name: str, temperature: float = 0.7, max_tokens: int = 1024) -> bool:
        """
        Initialize the LLM model for all functionality.
        
        Args:
            model_name: Name of the model to use
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_ollama_running():
            if not self.start_ollama():
                return False
        
        try:
            self.model_name = model_name
            self.temperature = temperature
            self.max_tokens = max_tokens
            
            self.model_initialized = True
            self.selected_model = model_name
            return True
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            return False
    
    def is_ollama_running(self) -> bool:
        """
        Check if Ollama is running locally.
        
        Returns:
            bool: True if Ollama is running, False otherwise
        """
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False
    
    def start_ollama(self) -> bool:
        """
        Start Ollama locally based on the platform.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            system = platform.system()
            print(f"Starting Ollama on {system}...")
            
            # Start Ollama based on platform
            if system == "Windows":
                # For Windows
                import subprocess
                subprocess.Popen(
                    ["ollama", "serve"], 
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                
            elif system == "Linux":
                # Try systemd first on Linux
                try:
                    import subprocess
                    subprocess.run(
                        ["systemctl", "--user", "start", "ollama"],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=5
                    )
                except (subprocess.SubprocessError, FileNotFoundError):
                    # Fallback to direct startup
                    import subprocess
                    subprocess.Popen(
                        ["ollama", "serve"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
            elif system == "Darwin":  # macOS
                import subprocess
                subprocess.Popen(
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
                if self.is_ollama_running():
                    print("Ollama started successfully!")
                    return True
                    
            print("Ollama did not start within expected time")
            return False
            
        except Exception as e:
            print(f"Error starting Ollama: {str(e)}")
            return False
    
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
    
    #
    # Code Generation Methods
    #
    
    def generate_exercise(self, difficulty: str = "medium", focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a new code review exercise.
        
        Args:
            difficulty: Difficulty level (easy, medium, hard)
            focus_areas: List of areas to focus on
            
        Returns:
            Dict with exercise data
        """
        if not self.model_initialized:
            return {"success": False, "error": "Model not initialized"}
        
        # Generate code with intentional issues
        code_snippet = self.generate_code_snippet(difficulty, focus_areas)
        
        # Compile the generated code to verify issues
        compile_results = self._compile_code(code_snippet)
        
        # Run checkstyle on the code
        checkstyle_results = self._run_checkstyle(code_snippet)
        
        # Analyze and store the ground truth of issues
        problem_analysis = self.analyze_code_issues(
            code_snippet, 
            compile_results, 
            checkstyle_results
        )
        
        # Generate expert review for later comparison
        expert_review = self.generate_expert_review(
            code_snippet,
            compile_results,
            checkstyle_results,
            problem_analysis
        )
        
        # Return the complete exercise package
        return {
            "success": True,
            "code_snippet": code_snippet,
            "compile_results": compile_results,
            "checkstyle_results": checkstyle_results,
            "problem_analysis": problem_analysis,
            "expert_review": expert_review,
            "difficulty": difficulty,
            "focus_areas": focus_areas or []
        }
    
    def generate_code_snippet(
        self, 
        difficulty: str = "medium", 
        focus_areas: Optional[List[str]] = None
    ) -> str:
        """
        Generate a Java code snippet with intentional issues for review.
        
        Args:
            difficulty: Difficulty level (easy, medium, hard)
            focus_areas: List of areas to focus on
            
        Returns:
            A Java code snippet as a string
        """
        # Default focus areas if none provided
        if not focus_areas:
            focus_areas = ["naming", "structure", "error_handling"]
        
        # Select specific errors to include based on difficulty and focus areas
        specific_errors = self._select_errors_for_snippet(difficulty, focus_areas)
        
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
        focus_areas: List[str]
    ) -> List[Dict[str, str]]:
        """
        Select specific errors to include in the code snippet.
        
        Args:
            difficulty: Difficulty level
            focus_areas: Areas to focus on
            
        Returns:
            List of selected errors to include
        """
        selected_errors = []
        
        # Determine number of errors based on difficulty
        if difficulty == "easy":
            num_errors = 3
        elif difficulty == "medium":
            num_errors = 4
        else:  # hard
            num_errors = 6
        
        # If error databases are loaded, use them to select specific errors
        if self.build_errors and self.checkstyle_errors:
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
                category = relevant_build_categories[_ % len(relevant_build_categories)]
                if category in self.build_errors:
                    error = self.build_errors[category][_ % len(self.build_errors[category])]
                    selected_errors.append({
                        "type": "build",
                        "category": category,
                        "name": error["error_name"],
                        "description": error["description"]
                    })
            
            # Select checkstyle errors
            for _ in range(num_checkstyle_errors):
                category = relevant_checkstyle_categories[_ % len(relevant_checkstyle_categories)]
                if category in self.checkstyle_errors:
                    error = self.checkstyle_errors[category][_ % len(self.checkstyle_errors[category])]
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
            
            # Select errors based on number needed, cycling through the list if necessary
            for i in range(num_errors):
                selected_errors.append(generic_errors[i % len(generic_errors)])
        
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
    
    #
    # Code Analysis Methods
    #
    
    def _compile_code(self, code_snippet: str) -> Dict[str, Any]:
        """
        Compile the Java code snippet.
        
        Args:
            code_snippet: Java code to compile
            
        Returns:
            Dict with compilation results
        """
        with tempfile.NamedTemporaryFile(suffix=".java", delete=False) as tmp_file:
            tmp_file.write(code_snippet.encode())
            tmp_path = tmp_file.name
        
        try:
            # Compile the Java code
            from modules.compiler import compile_java
            compile_results = compile_java(tmp_path)
            
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
                
            return compile_results
        except Exception as e:
            return {
                "success": False,
                "errors": f"Error compiling code: {str(e)}",
                "output": "",
                "command": ""
            }
    
    def _run_checkstyle(self, code_snippet: str) -> Dict[str, Any]:
        """
        Run checkstyle on the Java code snippet.
        
        Args:
            code_snippet: Java code to analyze
            
        Returns:
            Dict with checkstyle results
        """
        with tempfile.NamedTemporaryFile(suffix=".java", delete=False) as tmp_file:
            tmp_file.write(code_snippet.encode())
            tmp_path = tmp_file.name
        
        try:
            # Run checkstyle
            from modules.checkstyle import run_checkstyle
            checkstyle_results = run_checkstyle(tmp_path)
            
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
                
            return checkstyle_results
        except Exception as e:
            return {
                "success": False,
                "issues": [],
                "error": f"Error running checkstyle: {str(e)}",
                "command": ""
            }
    
    #
    # Problem Tracker Methods
    #
    
    def analyze_code_issues(
        self, 
        code_snippet: str, 
        compile_results: Dict[str, Any], 
        checkstyle_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a code snippet to identify all issues.
        
        Args:
            code_snippet: The Java code snippet
            compile_results: Results from compiling the code
            checkstyle_results: Results from running checkstyle
            
        Returns:
            Dict with analyzed issues data
        """
        # Generate a unique identifier for this code snippet
        snippet_id = self._generate_snippet_id(code_snippet)
        
        # Initialize issues list
        issues = []
        
        # Process compilation errors if any
        if not compile_results["success"]:
            compile_issues = self._analyze_compilation_errors(
                compile_results["errors"], 
                code_snippet
            )
            issues.extend(compile_issues)
        
        # Process checkstyle issues if any
        if checkstyle_results["issues"]:
            style_issues = self._analyze_checkstyle_issues(
                checkstyle_results["issues"], 
                code_snippet
            )
            issues.extend(style_issues)
        
        # Identify additional logical or semantic issues
        logical_issues = self._identify_logical_issues(code_snippet)
        issues.extend(logical_issues)
        
        # Store the analyzed issues in the database
        analysis_result = {
            "snippet_id": snippet_id,
            "issues": issues,
            "issue_count": len(issues),
            "categories": self._categorize_issues(issues),
            "difficulty_score": self._calculate_difficulty_score(issues)
        }
        
        self.issues_database[snippet_id] = analysis_result
        return analysis_result
    
    def _generate_snippet_id(self, code_snippet: str) -> str:
        """
        Generate a unique identifier for a code snippet.
        
        Args:
            code_snippet: The Java code snippet
            
        Returns:
            A unique string identifier
        """
        # Use a hash of the code to generate a unique ID
        hash_object = hashlib.md5(code_snippet.encode())
        return hash_object.hexdigest()[:12]
    
    def _analyze_compilation_errors(
        self, 
        error_output: str, 
        code_snippet: str
    ) -> List[Dict[str, Any]]:
        """
        Analyze compilation error output to identify specific issues.
        
        Args:
            error_output: Compilation error output
            code_snippet: The Java code snippet
            
        Returns:
            List of identified compilation issues
        """
        issues = []
        
        # Common regex patterns for error extraction
        error_patterns = [
            # Cannot find symbol
            (r"cannot find symbol\s*\n.*symbol\s*:\s*(\w+)\s*(\w+)(?:\s*\n.*location\s*:\s*(.*))?", 
             "Cannot find symbol"),
            
            # Incompatible types
            (r"incompatible types\s*:(?:\s*\n.*)?found\s*:\s*(\w+)(?:\s*\n.*)?required\s*:\s*(\w+)", 
             "Incompatible types"),
            
            # Missing return statement
            (r"missing return statement", 
             "Missing return statement"),
            
            # Unreported exception
            (r"unreported exception\s*(\w+)(?:\s*must be caught or declared to be thrown)?", 
             "Unreported exception"),
             
            # Class not found
            (r"class\s*(\w+)\s*not found", 
             "Class not found")
        ]
        
        # Extract error locations and types
        for pattern, error_type in error_patterns:
            matches = re.finditer(pattern, error_output, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                # Extract line number from error message if available
                line_match = re.search(r".*:(\d+):", error_output)
                line_num = int(line_match.group(1)) if line_match else None
                
                # Extract code context from the snippet if we have a line number
                code_context = None
                if line_num:
                    lines = code_snippet.split('\n')
                    if 0 <= line_num-1 < len(lines):
                        start_line = max(0, line_num-2)
                        end_line = min(len(lines), line_num+1)
                        code_context = '\n'.join(lines[start_line:end_line])
                
                # Look up additional error details if available
                error_details = None
                if self.build_errors:
                    for category in self.build_errors:
                        for error in self.build_errors[category]:
                            if error_type in error["error_name"]:
                                error_details = error["description"]
                                break
                        if error_details:
                            break
                
                issues.append({
                    "type": "compilation",
                    "error_type": error_type,
                    "line": line_num,
                    "description": error_details or "Compilation error",
                    "code_context": code_context,
                    "severity": "high",
                    "educational_value": "high",
                    "fix_difficulty": "medium"
                })
        
        # If no specific patterns matched but we have errors, add a generic entry
        if not issues and error_output:
            issues.append({
                "type": "compilation",
                "error_type": "Other compilation error",
                "description": "The code has compilation errors",
                "severity": "high",
                "educational_value": "medium",
                "fix_difficulty": "medium"
            })
        
        return issues
    
    def _analyze_checkstyle_issues(
        self, 
        style_issues: List[Dict[str, Any]], 
        code_snippet: str
    ) -> List[Dict[str, Any]]:
        """
        Analyze checkstyle issues to categorize and prioritize them.
        
        Args:
            style_issues: List of checkstyle issues
            code_snippet: The Java code snippet
            
        Returns:
            List of analyzed style issues
        """
        analyzed_issues = []
        
        # Lines of the code snippet for context
        lines = code_snippet.split('\n')
        
        for issue in style_issues:
            line_num = issue.get("line", 0)
            rule = issue.get("rule", "Unknown")
            message = issue.get("message", "Style issue")
            
            # Extract code context
            code_context = None
            if 0 < line_num <= len(lines):
                start_line = max(0, line_num-2)
                end_line = min(len(lines), line_num+1)
                code_context = '\n'.join(lines[start_line:end_line])
            
            # Determine category and significance
            category = self._determine_checkstyle_category(rule)
            severity = self._determine_issue_severity(category, rule)
            
            # Find detailed description from checkstyle_errors if available
            description = message
            if self.checkstyle_errors:
                for category_name, checks in self.checkstyle_errors.items():
                    for check in checks:
                        if check.get("check_name") == rule:
                            description = check.get("description", message)
                            break
            
            analyzed_issues.append({
                "type": "checkstyle",
                "error_type": rule,
                "category": category,
                "line": line_num,
                "description": description,
                "message": message,
                "code_context": code_context,
                "severity": severity,
                "educational_value": "medium" if severity == "high" else "low",
                "fix_difficulty": "low"
            })
        
        return analyzed_issues
    
    def _determine_checkstyle_category(self, rule: str) -> str:
        """
        Determine the category of a checkstyle rule.
        
        Args:
            rule: The checkstyle rule name
            
        Returns:
            Category name as a string
        """
        # If we have the checkstyle_errors database, look up the category
        if self.checkstyle_errors:
            for category, checks in self.checkstyle_errors.items():
                for check in checks:
                    if check.get("check_name") == rule:
                        return category
        
        # Otherwise use heuristics to determine category
        if rule in ["ConstantName", "LocalVariableName", "MemberName", "MethodName", 
                    "ParameterName", "StaticVariableName", "TypeName"]:
            return "NamingConventionChecks"
        elif rule in ["EmptyBlock", "NeedBraces", "LeftCurly", "RightCurly"]:
            return "BlockChecks"
        elif rule in ["WhitespaceAround", "WhitespaceAfter", "NoWhitespaceAfter", 
                      "NoWhitespaceBefore", "GenericWhitespace"]:
            return "WhitespaceAndFormattingChecks"
        elif rule in ["AvoidStarImport", "RedundantImport", "UnusedImports"]:
            return "ImportChecks"
        elif rule in ["JavadocMethod", "JavadocType", "JavadocVariable"]:
            return "JavadocChecks"
        elif rule in ["CyclomaticComplexity", "BooleanExpressionComplexity"]:
            return "MetricsChecks"
        elif rule in ["MagicNumber", "EmptyStatement", "SimplifyBooleanExpression"]:
            return "CodeQualityChecks"
        else:
            return "OtherStyleChecks"
    
    def _determine_issue_severity(self, category: str, rule: str) -> str:
        """
        Determine the severity of an issue based on its category and rule.
        
        Args:
            category: The issue category
            rule: The specific rule or error type
            
        Returns:
            Severity as 'high', 'medium', or 'low'
        """
        # High severity issues
        high_severity_categories = [
            "CompileTimeErrors", "RuntimeErrors", "LogicalErrors"
        ]
        
        high_severity_rules = [
            "EqualsHashCode", "StringLiteralEquality", "EmptyBlock",
            "MissingSwitchDefault", "FallThrough", "MultipleVariableDeclarations"
        ]
        
        # Medium severity issues
        medium_severity_categories = [
            "CodeQualityChecks", "ImportChecks", "MetricsChecks"
        ]
        
        medium_severity_rules = [
            "MagicNumber", "VisibilityModifier", "AvoidStarImport",
            "UnusedImports", "ParameterNumber", "MethodLength"
        ]
        
        # Determine severity based on category and rule
        if category in high_severity_categories or rule in high_severity_rules:
            return "high"
        elif category in medium_severity_categories or rule in medium_severity_rules:
            return "medium"
        else:
            return "low"
    
    def _identify_logical_issues(self, code_snippet: str) -> List[Dict[str, Any]]:
        """
        Identify potential logical issues in the code.
        
        Args:
            code_snippet: The Java code snippet
            
        Returns:
            List of identified logical issues
        """
        issues = []
        
        # Common logical issue patterns
        logical_patterns = [
            # Using == for string comparison instead of equals()
            (r'if\s*\(\s*(\w+)\s*==\s*"([^"]+)"\s*\)', 
             "String comparison using == instead of equals()",
             "Using == to compare strings checks for reference equality instead of content equality"),
            
            # Assignment in conditional
            (r'if\s*\(\s*(\w+)\s*=\s*([^=][^)]+)\s*\)', 
             "Using assignment instead of comparison",
             "Assignment operator used in conditional, likely meant to use == for comparison"),
            
            # Uninitialized variables
            (r'(\w+)\s+(\w+);\s*[^=]*\1\.|\w+\(.*\1', 
             "Variable might not have been initialized",
             "Variable is used before it may have been initialized"),
            
            # Empty catch blocks
            (r'catch\s*\(.*\)\s*\{\s*\}', 
             "Empty catch block",
             "Empty catch block swallows exceptions without handling them"),
            
            # Integer division leading to potential truncation
            (r'(int|long)\s+(\w+)\s*=\s*(\d+)\s*/\s*(\d+)', 
             "Unintended integer division",
             "Division between integers results in an integer, possible loss of precision")
        ]
        
        # Look for patterns in the code
        for pattern, issue_name, description in logical_patterns:
            matches = re.finditer(pattern, code_snippet)
            for match in matches:
                # Get the line number by counting newlines
                line_num = code_snippet[:match.start()].count('\n') + 1
                
                # Extract code context
                lines = code_snippet.split('\n')
                start_line = max(0, line_num-2)
                end_line = min(len(lines), line_num+1)
                code_context = '\n'.join(lines[start_line:end_line])
                
                issues.append({
                    "type": "logical",
                    "error_type": issue_name,
                    "line": line_num,
                    "description": description,
                    "code_context": code_context,
                    "severity": "high",
                    "educational_value": "high",
                    "fix_difficulty": "medium"
                })
        
        return issues
    
    def _categorize_issues(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Categorize issues by type and count.
        
        Args:
            issues: List of identified issues
            
        Returns:
            Dict with categories and counts
        """
        categories = {}
        
        # Count issues by type
        for issue in issues:
            issue_type = issue.get("type", "other")
            if issue_type == "checkstyle":
                category = issue.get("category", "other")
                categories[category] = categories.get(category, 0) + 1
            else:
                categories[issue_type] = categories.get(issue_type, 0) + 1
        
        # Count by severity
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        for issue in issues:
            severity = issue.get("severity", "low")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Add severity counts to categories
        categories.update(severity_counts)
        
        return categories
    
    def _calculate_difficulty_score(self, issues: List[Dict[str, Any]]) -> int:
        """
        Calculate an overall difficulty score for the issues.
        
        Args:
            issues: List of identified issues
            
        Returns:
            Difficulty score (1-10)
        """
        # Base score starts at 1
        score = 1
        
        # Add points based on severity and number of issues
        severity_weights = {"high": 2, "medium": 1, "low": 0.5}
        
        for issue in issues:
            severity = issue.get("severity", "low")
            weight = severity_weights.get(severity, 0.5)
            score += weight
        
        # Cap the score at 10
        return min(int(score), 10)
    
    #
    # Review Analysis Methods
    #
    
    def analyze_student_review(self, exercise_data: Dict[str, Any], student_review: str) -> Dict[str, Any]:
        """
        Analyze a student's code review against the expert review.
        
        Args:
            exercise_data: Exercise data from generate_exercise
            student_review: Student's review text
            
        Returns:
            Dict with analysis results
        """
        if not self.model_initialized:
            return {"success": False, "error": "Model not initialized"}
        
        # Compare reviews
        comparison = self.compare_reviews(
            exercise_data["expert_review"],
            student_review,
            exercise_data["problem_analysis"]
        )
        
        # Track student progress
        self.update_student_progress(
            comparison,
            exercise_data["difficulty"],
            exercise_data["focus_areas"]
        )
        
        # Generate personalized feedback based on comparison
        feedback = self.generate_feedback(
            comparison,
            exercise_data["problem_analysis"]
        )
        
        return {
            "success": True,
            "comparison": comparison,
            "feedback": feedback,
            "progress": self.get_current_progress()
        }
    
    def compare_reviews(
        self, 
        expert_review: str, 
        student_review: str,
        problem_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare a student's review with the expert review.
        
        Args:
            expert_review: Expert review of the code
            student_review: Student's review of the code
            problem_analysis: Analysis of the problems in the code
            
        Returns:
            Dict with comparison results
        """
        # First perform an NLP-driven comparison using the LLM
        nlp_comparison = self._perform_nlp_comparison(expert_review, student_review, problem_analysis)
        
        # Then perform a more structured analysis of which issues were caught
        caught_issues, missed_issues = self._analyze_caught_issues(student_review, problem_analysis)
        
        # Calculate metrics
        metrics = self._calculate_review_metrics(caught_issues, missed_issues, problem_analysis)
        
        # Combine results
        comparison_result = {
            "feedback": nlp_comparison.get("feedback", ""),
            "strengths": nlp_comparison.get("strengths", []),
            "improvements": nlp_comparison.get("improvements", []),
            "missed_issues": nlp_comparison.get("missed_issues", []),
            "caught_issues_count": len(caught_issues),
            "missed_issues_count": len(missed_issues),
            "caught_issues": caught_issues,
            "missed_issues": missed_issues,
            "metrics": metrics,
            "overall_score": metrics.get("overall_score", 0)
        }
        
        return comparison_result
    
    def _perform_nlp_comparison(
        self, 
        expert_review: str, 
        student_review: str,
        problem_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use NLP (via LLM) to compare the student and expert reviews.
        
        Args:
            expert_review: Expert review of the code
            student_review: Student's review of the code
            problem_analysis: Analysis of the problems in the code
            
        Returns:
            Dict with NLP comparison results
        """
        system_prompt = """
        You are an experienced programming instructor evaluating a student's code review skills.
        Compare the student's review with the expert AI review to provide constructive feedback.
        Focus on what the student did well and what they missed or could improve.
        Be encouraging but thorough in your assessment.
        """
        
        # Include information about the known issues in the code
        issues_summary = "\n".join([
            f"- {i+1}. {issue.get('error_type', 'Unknown')} ({issue.get('severity', 'medium')} severity): {issue.get('description', 'No description')}"
            for i, issue in enumerate(problem_analysis.get("issues", []))
        ])
        
        prompt = f"""
        Compare the following student code review with the expert AI review.
        
        EXPERT REVIEW:
        {expert_review}
        
        STUDENT REVIEW:
        {student_review}
        
        KNOWN ISSUES IN THE CODE:
        {issues_summary}
        
        Analyze the comparison and provide:
        1. Overall feedback on the student's review quality
        2. A list of strengths in the student's review
        3. A list of areas for improvement
        4. A list of important issues that the student missed but were caught in the expert review
        
        Return your analysis in JSON format with the following structure:
        {{
            "feedback": "Overall feedback...",
            "strengths": ["Strength 1", "Strength 2", ...],
            "improvements": ["Improvement 1", "Improvement 2", ...],
            "missed_issues": ["Missed issue 1", "Missed issue 2", ...]
        }}
        """
        
        response = self._call_ollama(prompt, system_prompt)
        
        # Try to parse the response as JSON
        try:
            # Find JSON content in the response
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
            return {
                "feedback": response,
                "strengths": ["Your review was received"],
                "improvements": ["Try to be more specific in your feedback"],
                "missed_issues": ["Unable to analyze missed issues"]
            }
    
    def _analyze_caught_issues(
        self, 
        student_review: str,
        problem_analysis: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Analyze which issues were caught or missed by the student.
        
        Args:
            student_review: Student's review of the code
            problem_analysis: Analysis of the problems in the code
            
        Returns:
            Tuple of (caught_issues, missed_issues)
        """
        caught_issues = []
        missed_issues = []
        
        # Get all known issues from the problem analysis
        known_issues = problem_analysis.get("issues", [])
        
        # For each known issue, check if it was mentioned in the student review
        for issue in known_issues:
            issue_type = issue.get("error_type", "")
            description = issue.get("description", "")
            line = issue.get("line", 0)
            code_context = issue.get("code_context", "")
            
            # Create various ways the issue might be mentioned
            search_terms = [
                issue_type.lower(),
                # Extract key terms from description
                *[term.lower() for term in re.findall(r'\b\w{4,}\b', description)]
            ]
            
            # If we have line numbers, look for those too
            if line > 0:
                search_terms.append(f"line {line}")
                search_terms.append(f"Line {line}")
            
            # Check if any search terms appear in the student review
            issue_mentioned = False
            for term in search_terms:
                if term and term in student_review.lower():
                    issue_mentioned = True
                    break
            
            # If code context is available, check if it's mentioned
            if code_context and not issue_mentioned:
                # Extract code snippets from context
                code_snippets = re.findall(r'\b\w+\b', code_context)
                for snippet in code_snippets:
                    if len(snippet) > 3 and snippet in student_review:  # Avoid short variable names like 'i'
                        issue_mentioned = True
                        break
            
            # Add to appropriate list
            if issue_mentioned:
                caught_issues.append(issue)
            else:
                missed_issues.append(issue)
        
        return caught_issues, missed_issues
    
    def _calculate_review_metrics(
        self, 
        caught_issues: List[Dict[str, Any]], 
        missed_issues: List[Dict[str, Any]],
        problem_analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate metrics for the review performance.
        
        Args:
            caught_issues: Issues caught by the student
            missed_issues: Issues missed by the student
            problem_analysis: Analysis of the problems in the code
            
        Returns:
            Dict with various metrics
        """
        # Calculate total issues and percentages
        total_issues = len(caught_issues) + len(missed_issues)
        
        if total_issues == 0:
            return {
                "coverage": 0.0,
                "precision": 0.0,
                "severity_weighted_coverage": 0.0,
                "overall_score": 0.0
            }
        
        # Basic coverage: What percentage of issues were caught
        coverage = len(caught_issues) / total_issues if total_issues > 0 else 0.0
        
        # Severity-weighted coverage: Catching high-severity issues is more important
        severity_weights = {"high": 3.0, "medium": 2.0, "low": 1.0}
        
        severity_sum = sum(severity_weights.get(issue.get("severity", "low"), 1.0) 
                           for issue in caught_issues + missed_issues)
                           
        caught_severity_sum = sum(severity_weights.get(issue.get("severity", "low"), 1.0) 
                                 for issue in caught_issues)
        
        severity_weighted_coverage = caught_severity_sum / severity_sum if severity_sum > 0 else 0.0
        
        # Overall score: 0-100 scale
        overall_score = int((coverage * 0.4 + severity_weighted_coverage * 0.6) * 100)
        
        # Categorize the issues caught/missed by type
        caught_by_type = {}
        missed_by_type = {}
        
        for issue in caught_issues:
            issue_type = issue.get("type", "other")
            caught_by_type[issue_type] = caught_by_type.get(issue_type, 0) + 1
            
        for issue in missed_issues:
            issue_type = issue.get("type", "other")
            missed_by_type[issue_type] = missed_by_type.get(issue_type, 0) + 1
        
        # Return all metrics
        return {
            "coverage": round(coverage, 2),
            "severity_weighted_coverage": round(severity_weighted_coverage, 2),
            "overall_score": overall_score,
            "caught_by_type": caught_by_type,
            "missed_by_type": missed_by_type
        }
    
    #
    # Feedback Generation Methods
    #
    
    def generate_feedback(
        self, 
        comparison_result: Dict[str, Any],
        problem_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate personalized feedback based on the review comparison.
        
        Args:
            comparison_result: Result of comparing student and expert reviews
            problem_analysis: Analysis of the problems in the code
            
        Returns:
            Dict with personalized feedback
        """
        # Combine template-based and LLM-generated feedback
        template_feedback = self._generate_template_feedback(
            comparison_result, 
            problem_analysis
        )
        
        # Use LLM to generate more personalized feedback
        llm_feedback = self._generate_llm_feedback(
            comparison_result, 
            problem_analysis
        )
        
        # Combine feedback components
        detailed_feedback = {
            "summary": llm_feedback.get("summary", ""),
            "strengths": comparison_result.get("strengths", []),
            "improvements": comparison_result.get("improvements", []),
            "missed_issues_feedback": llm_feedback.get("missed_issues_feedback", []),
            "learning_tips": llm_feedback.get("learning_tips", []),
            "next_steps": llm_feedback.get("next_steps", ""),
            "templates": template_feedback
        }
        
        return detailed_feedback
    
    def _generate_template_feedback(
        self, 
        comparison_result: Dict[str, Any],
        problem_analysis: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Generate feedback using templates.
        
        Args:
            comparison_result: Result of comparing student and expert reviews
            problem_analysis: Analysis of the problems in the code
            
        Returns:
            Dict with template-based feedback
        """
        template_feedback = {
            "positive": [],
            "missed_high": [],
            "missed_medium": [],
            "explanations": [],
            "improvements": []
        }
        
        # Generate positive feedback for caught issues
        caught_issues = comparison_result.get("caught_issues", [])
        for issue in caught_issues[:3]:  # Limit to 3 positive comments
            issue_type = issue.get("error_type", "issue")
            template = random.choice(self.templates["positive"])
            
            # Fill in template variables
            feedback = template.format(
                issue_type=issue_type,
                focus=self._get_focus_from_issue(issue),
                reason=self._get_reason_from_issue(issue),
                concept=self._get_concept_from_issue_type(issue_type)
            )
            
            template_feedback["positive"].append(feedback)
        
        # Generate feedback for missed high-severity issues
        missed_issues = comparison_result.get("missed_issues", [])
        high_severity_missed = [i for i in missed_issues if i.get("severity") == "high"]
        
        for issue in high_severity_missed[:3]:  # Limit to 3 high-severity issues
            issue_type = issue.get("error_type", "issue")
            line = issue.get("line", "?")
            template = random.choice(self.templates["missed_high"])
            
            # Fill in template variables
            feedback = template.format(
                issue_type=issue_type,
                issue_description=issue.get("description", "issue"),
                line=line,
                reason=self._get_reason_from_issue(issue),
                consequence=self._get_consequence_from_issue(issue)
            )
            
            template_feedback["missed_high"].append(feedback)
        
        # Generate feedback for missed medium-severity issues
        medium_severity_missed = [i for i in missed_issues if i.get("severity") == "medium"]
        
        for issue in medium_severity_missed[:2]:  # Limit to 2 medium-severity issues
            issue_type = issue.get("error_type", "issue")
            line = issue.get("line", "?")
            template = random.choice(self.templates["missed_medium"])
            
            # Fill in template variables
            feedback = template.format(
                issue_type=issue_type,
                issue_description=issue.get("description", "issue"),
                line=line
            )
            
            template_feedback["missed_medium"].append(feedback)
        
        # Generate explanations for the most important missed issues
        important_missed = sorted(
            missed_issues, 
            key=lambda x: 3 if x.get("severity") == "high" else (2 if x.get("severity") == "medium" else 1),
            reverse=True
        )
        
        for issue in important_missed[:3]:  # Limit to 3 explanations
            issue_type = issue.get("error_type", "issue")
            template = random.choice(self.templates["explanation"])
            
            # Fill in template variables
            feedback = template.format(
                issue_type=issue_type,
                issue_description=issue.get("description", "issue"),
                pattern=self._get_pattern_from_issue(issue),
                reason=self._get_reason_from_issue(issue),
                explanation=self._get_explanation_from_issue(issue),
                solution=self._get_solution_from_issue(issue)
            )
            
            template_feedback["explanations"].append(feedback)
        
        # Generate improvement suggestions
        improvement_areas = self._identify_improvement_areas(comparison_result)
        
        for area in improvement_areas[:3]:  # Limit to 3 improvement suggestions
            template = random.choice(self.templates["improvement"])
            
            # Fill in template variables
            feedback = template.format(
                issue_type=area.get("issue_type", "issue"),
                technique=area.get("technique", "reviewing more carefully"),
                approach=area.get("approach", "a more systematic approach"),
                strategy=area.get("strategy", "checking carefully")
            )
            
            template_feedback["improvements"].append(feedback)
        
        return template_feedback
    
    def _generate_llm_feedback(
        self, 
        comparison_result: Dict[str, Any],
        problem_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate personalized feedback using the LLM.
        
        Args:
            comparison_result: Result of comparing student and expert reviews
            problem_analysis: Analysis of the problems in the code
            
        Returns:
            Dict with LLM-generated feedback
        """
        system_prompt = """
        You are an expert programming instructor providing personalized feedback to a student on their code review skills.
        Your feedback should be encouraging, educational, and specific.
        Focus on helping the student improve their review skills by explaining concepts they missed and suggesting concrete techniques.
        Format your response in JSON as specified in the prompt.
        """
        
        # Prepare summary of caught and missed issues
        caught_issues_summary = "\n".join([
            f"- {i+1}. {issue.get('error_type', 'Unknown')} ({issue.get('severity', 'medium')} severity): {issue.get('description', 'No description')}"
            for i, issue in enumerate(comparison_result.get("caught_issues", []))
        ])
        
        missed_issues_summary = "\n".join([
            f"- {i+1}. {issue.get('error_type', 'Unknown')} ({issue.get('severity', 'medium')} severity): {issue.get('description', 'No description')}"
            for i, issue in enumerate(comparison_result.get("missed_issues", []))
        ])
        
        # Include learning history if available
        learning_history_summary = ""
        if self.learning_history["completed_exercises"] > 0:
            # Extract relevant information from learning history
            strengths = self.learning_history.get("strengths", [])
            weaknesses = self.learning_history.get("weaknesses", [])
            completed_exercises = self.learning_history.get("completed_exercises", 0)
            
            learning_history_summary = f"""
            Student learning history:
            - Completed exercises: {completed_exercises}
            - Strengths: {', '.join(strengths) if strengths else 'None identified yet'}
            - Areas for improvement: {', '.join(weaknesses) if weaknesses else 'None identified yet'}
            """
        
        prompt = f"""
        Generate personalized feedback for a student's code review.
        
        Review comparison results:
        - Overall score: {comparison_result.get('overall_score', 0)}/100
        - Issues caught: {len(comparison_result.get('caught_issues', []))}
        - Issues missed: {len(comparison_result.get('missed_issues', []))}
        
        Issues the student caught:
        {caught_issues_summary if caught_issues_summary else "None"}
        
        Issues the student missed:
        {missed_issues_summary if missed_issues_summary else "None"}
        
        {learning_history_summary}
        
        Please provide:
        1. A brief summary of the student's review performance
        2. Specific feedback on the most important missed issues (prioritize high severity)
        3. Learning tips to help the student improve
        4. Suggested next steps for continued learning
        
        Return your feedback in JSON format with the following structure:
        {{
            "summary": "Brief overall assessment of the student's performance",
            "missed_issues_feedback": [
                "Feedback on missed issue 1",
                "Feedback on missed issue 2",
                ...
            ],
            "learning_tips": [
                "Learning tip 1",
                "Learning tip 2",
                ...
            ],
            "next_steps": "Suggested next steps for the student"
        }}
        """
        
        response = self._call_ollama(prompt, system_prompt)
        
        # Try to parse the response as JSON
        try:
            # Find JSON content in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                return result
            else:
                # Fallback if proper JSON is not found
                return {
                    "summary": "Your code review shows some good points and some areas for improvement.",
                    "missed_issues_feedback": ["Try to be more thorough in your reviews."],
                    "learning_tips": ["Practice identifying common Java coding issues."],
                    "next_steps": "Continue practicing code reviews with a focus on thoroughness."
                }
        except json.JSONDecodeError:
            # Handle the case where response is not valid JSON
            return {
                "summary": "Your code review shows partial understanding of the issues in the code.",
                "missed_issues_feedback": ["Some important issues were missed in your review."],
                "learning_tips": ["Try to systematically check for different categories of issues."],
                "next_steps": "Review Java best practices and common coding errors."
            }
    
    def _get_focus_from_issue(self, issue: Dict[str, Any]) -> str:
        """Extract a focus area description from an issue."""
        issue_type = issue.get("type", "")
        error_type = issue.get("error_type", "")
        
        if issue_type == "compilation":
            return "compilation errors"
        elif issue_type == "checkstyle":
            if "Name" in error_type:
                return "naming conventions"
            elif "Whitespace" in error_type:
                return "code formatting"
            else:
                return "code style"
        elif issue_type == "logical":
            return "logical correctness"
        else:
            return "code quality"
    
    def _get_reason_from_issue(self, issue: Dict[str, Any]) -> str:
        """Extract a reason why the issue is important."""
        description = issue.get("description", "")
        severity = issue.get("severity", "medium")
        
        if description:
            return description
        
        # Generic reasons based on severity
        if severity == "high":
            return "it could lead to runtime errors or incorrect behavior"
        elif severity == "medium":
            return "it impacts code readability and maintainability"
        else:
            return "it's a good practice to follow coding standards"
    
    def _get_concept_from_issue_type(self, issue_type: str) -> str:
        """Map an issue type to a programming concept."""
        concept_map = {
            "Cannot find symbol": "variable scope and declaration",
            "Incompatible types": "Java type system",
            "Missing return statement": "method control flow",
            "Unreported exception": "exception handling",
            "ConstantName": "Java naming conventions",
            "WhitespaceAround": "code formatting",
            "EmptyBlock": "error handling and code structure",
            "MagicNumber": "code maintainability",
            "String comparison using ==": "string handling in Java",
            "Using assignment instead of comparison": "conditional expressions"
        }
        
        return concept_map.get(issue_type, "code quality principles")
    
    def _get_pattern_from_issue(self, issue: Dict[str, Any]) -> str:
        """Extract a pattern that indicates this type of issue."""
        issue_type = issue.get("error_type", "")
        code_context = issue.get("code_context", "")
        
        # Return code context if available
        if code_context:
            return f"code like `{code_context.strip()}`"
        
        # Generic patterns based on issue type
        pattern_map = {
            "Cannot find symbol": "undefined variables or methods",
            "Incompatible types": "assigning values of one type to variables of another type",
            "Missing return statement": "methods with a return type but no return statement",
            "ConstantName": "constants not in UPPER_CASE",
            "WhitespaceAround": "missing spaces around operators or braces",
            "EmptyBlock": "empty catch blocks or if statements",
            "MagicNumber": "literal numbers in code instead of named constants",
            "String comparison using ==": "strings compared with == instead of equals()",
            "Using assignment instead of comparison": "single = in conditional statements"
        }
        
        return pattern_map.get(issue_type, "problematic code patterns")
    
    def _get_consequence_from_issue(self, issue: Dict[str, Any]) -> str:
        """Describe the potential consequence of an issue."""
        issue_type = issue.get("type", "")
        error_type = issue.get("error_type", "")
        
        # Consequences based on issue type
        if issue_type == "compilation":
            return "code that won't compile"
        elif issue_type == "logical":
            if "String comparison" in error_type:
                return "string comparisons failing unexpectedly"
            elif "assignment" in error_type:
                return "conditionals always evaluating to the assigned value"
            else:
                return "unexpected runtime behavior"
        elif issue_type == "checkstyle":
            return "code that's harder to maintain and understand"
        else:
            return "potential bugs or maintenance issues"
    
    def _get_explanation_from_issue(self, issue: Dict[str, Any]) -> str:
        """Provide an explanation of why the issue occurs."""
        # Use the description if available
        description = issue.get("description", "")
        if description:
            return description
        
        # Otherwise generate a generic explanation
        issue_type = issue.get("error_type", "")
        explanation_map = {
            "Cannot find symbol": "a variable or method is being used without being properly declared in scope",
            "Incompatible types": "Java can't automatically convert between these types",
            "Missing return statement": "Java requires all code paths in non-void methods to return a value",
            "String comparison using ==": "== checks if two strings are the same object in memory, not if they have the same content",
            "Using assignment instead of comparison": "= assigns a value whereas == compares values"
        }
        
        return explanation_map.get(issue_type, "the code violates Java conventions or best practices")
    
    def _get_solution_from_issue(self, issue: Dict[str, Any]) -> str:
        """Provide a solution for the issue."""
        issue_type = issue.get("error_type", "")
        
        # Solutions based on issue type
        solution_map = {
            "Cannot find symbol": "ensure the variable is declared before use and is in scope",
            "Incompatible types": "use proper casting or conversion methods",
            "Missing return statement": "add a return statement for all possible execution paths",
            "String comparison using ==": "use equals() method instead of == for string comparison",
            "Using assignment instead of comparison": "use == for comparison instead of =",
            "ConstantName": "use UPPER_CASE for constant names",
            "WhitespaceAround": "add appropriate spaces around operators and braces",
            "EmptyBlock": "either add meaningful code to the block or comment why it's intentionally empty",
            "MagicNumber": "replace literal numbers with named constants"
        }
        
        return solution_map.get(issue_type, "follow Java conventions and best practices")
    
    def _identify_improvement_areas(self, comparison_result: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Identify areas where the student can improve.
        
        Args:
            comparison_result: Result of comparing student and expert reviews
            
        Returns:
            List of improvement areas with suggestions
        """
        improvement_areas = []
        
        # Analyze missed issues by type
        missed_by_type = {}
        for issue in comparison_result.get("missed_issues", []):
            issue_type = issue.get("type", "other")
            if issue_type not in missed_by_type:
                missed_by_type[issue_type] = []
            missed_by_type[issue_type].append(issue)
        
        # For each type with multiple missed issues, add an improvement area
        for issue_type, issues in missed_by_type.items():
            if len(issues) >= 2:
                if issue_type == "compilation":
                    improvement_areas.append({
                        "issue_type": "compilation errors",
                        "technique": "checking for common Java syntax and semantic errors",
                        "approach": "reviewing the code with compilation errors in mind, thinking about variable scope, types, and method signatures",
                        "strategy": "mentally trace variable declarations and usages"
                    })
                elif issue_type == "checkstyle":
                    improvement_areas.append({
                        "issue_type": "code style issues",
                        "technique": "paying attention to naming conventions and code formatting",
                        "approach": "scanning the code for style inconsistencies, especially in variable names and whitespace",
                        "strategy": "check variable names against Java naming conventions"
                    })
                elif issue_type == "logical":
                    improvement_areas.append({
                        "issue_type": "logical errors",
                        "technique": "mentally executing the code to find logic flaws",
                        "approach": "looking for common logical mistakes like using = instead of == or string comparison issues",
                        "strategy": "test conditional statements by considering different input values"
                    })
        
        # Add general improvement suggestions if needed
        if not improvement_areas:
            improvement_areas.append({
                "issue_type": "general issues",
                "technique": "using a systematic approach to code review",
                "approach": "going through the code line by line with a checklist of common issues",
                "strategy": "use a code review checklist covering different types of potential issues"
            })
        
        return improvement_areas
    
    #
    # Progress Tracking Methods
    #
    
    def update_student_progress(
        self, 
        comparison_result: Dict[str, Any],
        difficulty: str,
        focus_areas: Optional[List[str]] = None
    ) -> None:
        """
        Update student progress based on review comparison results.
        
        Args:
            comparison_result: Result of comparing student and expert reviews
            difficulty: Difficulty level of the exercise
            focus_areas: Focus areas of the exercise
        """
        # Extract metrics from the comparison
        metrics = comparison_result.get("metrics", {})
        overall_score = metrics.get("overall_score", 0)
        caught_issues_count = comparison_result.get("caught_issues_count", 0)
        missed_issues_count = comparison_result.get("missed_issues_count", 0)
        caught_by_type = metrics.get("caught_by_type", {})
        missed_by_type = metrics.get("missed_by_type", {})
        
        # Create an exercise record
        exercise = {
            "timestamp": time.time(),
            "difficulty": difficulty,
            "focus_areas": focus_areas or [],
            "overall_score": overall_score,
            "caught_issues_count": caught_issues_count,
            "missed_issues_count": missed_issues_count,
            "caught_by_type": caught_by_type,
            "missed_by_type": missed_by_type
        }
        
        # Add to history
        self.learning_history["exercises"].append(exercise)
        self.learning_history["completed_exercises"] += 1
        self.learning_history["last_updated"] = time.time()
        
        # Update performance metrics
        self._update_performance_metrics(exercise)
        
        # Update strengths and weaknesses
        self._update_strengths_and_weaknesses()
        
        # Update student level
        self._update_student_level()
    
    def _update_performance_metrics(self, exercise: Dict[str, Any]) -> None:
        """
        Update performance metrics based on exercise results.
        
        Args:
            exercise: The completed exercise data
        """
        # Track performance by difficulty
        difficulty = exercise["difficulty"]
        if difficulty not in self.learning_history["performance"]:
            self.learning_history["performance"][difficulty] = {
                "exercises": 0,
                "average_score": 0,
                "total_caught": 0,
                "total_missed": 0
            }
        
        # Update difficulty stats
        performance = self.learning_history["performance"][difficulty]
        performance["exercises"] += 1
        performance["total_caught"] += exercise["caught_issues_count"]
        performance["total_missed"] += exercise["missed_issues_count"]
        
        # Recalculate average score
        total_score = performance["average_score"] * (performance["exercises"] - 1)
        total_score += exercise["overall_score"]
        performance["average_score"] = total_score / performance["exercises"]
        
        # Track performance by issue type
        for issue_type, count in exercise["caught_by_type"].items():
            category = f"{issue_type}_issues"
            if category not in self.learning_history["progress_by_category"]:
                self.learning_history["progress_by_category"][category] = []
            
            self.learning_history["progress_by_category"][category].append({
                "timestamp": exercise["timestamp"],
                "caught": count,
                "missed": exercise["missed_by_type"].get(issue_type, 0)
            })
    
    def _update_strengths_and_weaknesses(self) -> None:
        """Update student strengths and weaknesses based on accumulated data."""
        # Reset strengths and weaknesses
        self.learning_history["strengths"] = []
        self.learning_history["weaknesses"] = []
        
        # Look at the last 3 exercises for recent trends
        recent_exercises = self.learning_history["exercises"][-3:]
        if not recent_exercises:
            return
        
        # Aggregate issue types across recent exercises
        issue_performance = defaultdict(lambda: {"caught": 0, "missed": 0})
        
        for exercise in recent_exercises:
            for issue_type, count in exercise["caught_by_type"].items():
                issue_performance[issue_type]["caught"] += count
            
            for issue_type, count in exercise["missed_by_type"].items():
                issue_performance[issue_type]["missed"] += count
        
        # Identify strengths (high catch rate)
        for issue_type, counts in issue_performance.items():
            total = counts["caught"] + counts["missed"]
            if total > 0:
                catch_rate = counts["caught"] / total
                
                if catch_rate >= 0.8 and total >= 3:
                    self.learning_history["strengths"].append(issue_type)
                elif catch_rate <= 0.4 and total >= 3:
                    self.learning_history["weaknesses"].append(issue_type)
        
        # Add difficulty-based strengths/weaknesses
        for difficulty, performance in self.learning_history["performance"].items():
            if performance["exercises"] >= 2:
                if performance["average_score"] >= 80:
                    self.learning_history["strengths"].append(f"{difficulty} difficulty exercises")
                elif performance["average_score"] <= 50:
                    self.learning_history["weaknesses"].append(f"{difficulty} difficulty exercises")
    
    def _update_student_level(self) -> None:
        """Update the student's current level based on performance."""
        # Get performance at each difficulty level
        easy_perf = self.learning_history["performance"].get("easy", {"average_score": 0, "exercises": 0})
        medium_perf = self.learning_history["performance"].get("medium", {"average_score": 0, "exercises": 0})
        hard_perf = self.learning_history["performance"].get("hard", {"average_score": 0, "exercises": 0})
        
        # Determine level based on performance
        if hard_perf["exercises"] >= 3 and hard_perf["average_score"] >= 70:
            self.learning_history["current_level"] = "expert"
        elif medium_perf["exercises"] >= 3 and medium_perf["average_score"] >= 70:
            self.learning_history["current_level"] = "advanced"
        elif easy_perf["exercises"] >= 3 and easy_perf["average_score"] >= 70:
            self.learning_history["current_level"] = "intermediate"
        else:
            self.learning_history["current_level"] = "beginner"
    
    def get_learning_history(self) -> Dict[str, Any]:
        """
        Get the student's learning history.
        
        Returns:
            Dict with learning history
        """
        return self.learning_history
    
    def get_current_progress(self) -> Dict[str, Any]:
        """
        Get the student's current progress summary.
        
        Returns:
            Dict with progress summary
        """
        # Calculate recent trend (last 3 exercises)
        recent_scores = [ex["overall_score"] for ex in self.learning_history["exercises"][-3:]]
        
        trend = 0  # neutral
        if len(recent_scores) >= 2:
            trend = 1 if recent_scores[-1] > recent_scores[0] else (-1 if recent_scores[-1] < recent_scores[0] else 0)
        
        return {
            "level": self.learning_history["current_level"],
            "completed_exercises": self.learning_history["completed_exercises"],
            "strengths": self.learning_history["strengths"],
            "weaknesses": self.learning_history["weaknesses"],
            "recent_trend": trend,
            "last_score": recent_scores[-1] if recent_scores else 0
        }
    
    def recommend_next_exercise(self) -> Dict[str, Any]:
        """
        Recommend the next exercise based on student progress.
        
        Returns:
            Dict with recommendation data
        """
        # Default recommendation for beginners
        if self.learning_history["completed_exercises"] == 0:
            return {
                "difficulty": "easy",
                "focus_areas": ["naming", "structure"],
                "rationale": "Starting with the basics of code review"
            }
        
        # Decide on difficulty
        current_level = self.learning_history["current_level"]
        difficulty_mapping = {
            "beginner": "easy",
            "intermediate": "medium",
            "advanced": "medium",  # occasionally give hard exercises
            "expert": "hard"
        }
        
        recommended_difficulty = difficulty_mapping.get(current_level, "easy")
        
        # Occasionally increase difficulty to challenge the student
        # or decrease it if they're struggling
        last_exercise = self.learning_history["exercises"][-1]
        last_score = last_exercise["overall_score"]
        
        if recommended_difficulty == "easy" and last_score >= 80:
            recommended_difficulty = "medium"
            difficulty_rationale = "Increasing difficulty based on your good performance"
        elif recommended_difficulty == "medium" and last_score >= 85:
            recommended_difficulty = "hard"
            difficulty_rationale = "Challenging you with a harder exercise based on your excellent performance"
        elif recommended_difficulty == "medium" and last_score <= 40:
            recommended_difficulty = "easy"
            difficulty_rationale = "Providing an easier exercise to build confidence"
        elif recommended_difficulty == "hard" and last_score <= 50:
            recommended_difficulty = "medium"
            difficulty_rationale = "Adjusting difficulty to help reinforce concepts"
        else:
            difficulty_rationale = f"Continuing with {recommended_difficulty} difficulty exercises"
        
        # Decide on focus areas based on weaknesses
        recommended_focus = []
        
        # Map issue types to focus areas
        issue_to_focus = {
            "compilation": ["error_handling", "structure"],
            "checkstyle": ["naming", "structure"],
            "logical": ["logic", "error_handling"],
            "performance": ["performance", "algorithms"],
            "security": ["security", "error_handling"]
        }
        
        # Include weaknesses as focus areas
        for weakness in self.learning_history["weaknesses"][:2]:  # Limit to 2 weaknesses
            for issue_type, focus_areas in issue_to_focus.items():
                if issue_type in weakness.lower():
                    recommended_focus.extend(focus_areas)
        
        # If no specific weaknesses, provide balanced focus
        if not recommended_focus:
            # Include at least one strength area to maintain confidence
            if self.learning_history["strengths"]:
                strength_area = self.learning_history["strengths"][0]
                for issue_type, focus_areas in issue_to_focus.items():
                    if issue_type in strength_area.lower():
                        recommended_focus.append(focus_areas[0])
            
            # Add general focus areas
            general_areas = ["naming", "structure", "error_handling"]
            for area in general_areas:
                if area not in recommended_focus:
                    recommended_focus.append(area)
        
        # Ensure no duplicates and limit to 3 focus areas
        recommended_focus = list(set(recommended_focus))[:3]
        
        # Generate rationale
        if self.learning_history["weaknesses"]:
            focus_rationale = f"Focusing on {', '.join(recommended_focus)} to address areas needing improvement"
        else:
            focus_rationale = f"Providing a balanced focus on {', '.join(recommended_focus)}"
        
        return {
            "difficulty": recommended_difficulty,
            "focus_areas": recommended_focus,
            "rationale": f"{difficulty_rationale}. {focus_rationale}."
        }
    
    def generate_learning_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of learning progress.
        
        Returns:
            Dict with learning summary data
        """
        if not self.learning_history["exercises"]:
            return {
                "status": "No exercises completed yet",
                "recommendations": "Start with basic code review exercises"
            }
        
        # Calculate overall statistics
        total_caught = sum(ex["caught_issues_count"] for ex in self.learning_history["exercises"])
        total_missed = sum(ex["missed_issues_count"] for ex in self.learning_history["exercises"])
        total_issues = total_caught + total_missed
        
        catch_rate = total_caught / total_issues if total_issues > 0 else 0
        average_score = sum(ex["overall_score"] for ex in self.learning_history["exercises"]) / len(self.learning_history["exercises"])
        
        # Calculate progress over time
        scores_over_time = [
            {"timestamp": ex["timestamp"], "score": ex["overall_score"]}
            for ex in self.learning_history["exercises"]
        ]
        
        # Generate text summary
        if self.learning_history["completed_exercises"] <= 3:
            progress_summary = "You're just getting started with code reviews. Keep practicing to develop your skills."
        else:
            progress_summary = f"You've completed {self.learning_history['completed_exercises']} exercises and are at the {self.learning_history['current_level']} level."
            
            # Add trend information
            recent_scores = [ex["overall_score"] for ex in self.learning_history["exercises"][-3:]]
            if len(recent_scores) >= 3:
                if recent_scores[2] > recent_scores[0]:
                    progress_summary += " Your scores are improving!"
                elif recent_scores[2] < recent_scores[0]:
                    progress_summary += " Your recent scores have decreased slightly."
                else:
                    progress_summary += " Your scores have been consistent recently."
        
        # Generate recommendations
        if self.learning_history["weaknesses"]:
            recommendations = f"Focus on improving your {', '.join(self.learning_history['weaknesses'])}."
        else:
            recommendations = "Continue to practice a variety of code review exercises to maintain your skills."
        
        # Add next steps based on level
        if self.learning_history["current_level"] == "beginner":
            next_steps = "Practice identifying common Java errors and style issues."
        elif self.learning_history["current_level"] == "intermediate":
            next_steps = "Work on identifying subtle logical errors and performance issues."
        elif self.learning_history["current_level"] == "advanced":
            next_steps = "Challenge yourself with complex code reviews and focus on security concerns."
        else:  # expert
            next_steps = "Consider reviewing real-world code in open source projects to further refine your skills."
        
        return {
            "status": progress_summary,
            "level": self.learning_history["current_level"],
            "completed_exercises": self.learning_history["completed_exercises"],
            "strengths": self.learning_history["strengths"],
            "weaknesses": self.learning_history["weaknesses"],
            "average_score": round(average_score, 1),
            "catch_rate": round(catch_rate * 100, 1),
            "scores_over_time": scores_over_time,
            "recommendations": recommendations,
            "next_steps": next_steps
        }