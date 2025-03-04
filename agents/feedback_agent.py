"""
Feedback Agent for Peer Code Review Tutorial System

This agent is responsible for generating personalized, educational feedback
for students based on their code review performance.
"""

import json
import os
import random
import requests
from typing import Dict, List, Any, Optional
import backoff

class FeedbackAgent:
    """
    Feedback Agent that generates personalized educational feedback.
    
    This agent:
    1. Creates targeted feedback on missed issues
    2. Provides explanations tailored to the student's level
    3. Generates constructive suggestions for improvement
    4. Adjusts feedback style based on learning history
    """
    
    def __init__(self):
        """Initialize the Feedback Agent."""
        self.model_name = None
        self.temperature = 0.7
        self.max_tokens = 1024
        self.ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        
        # Define feedback templates for different scenarios
        self._load_feedback_templates()
    
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
    
    def generate_feedback(
        self, 
        comparison_result: Dict[str, Any],
        problem_analysis: Dict[str, Any],
        learning_history: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate personalized feedback based on the review comparison.
        
        Args:
            comparison_result: Result of comparing student and expert reviews
            problem_analysis: Analysis of the problems in the code
            learning_history: Optional history of student learning progress
            
        Returns:
            Dict with personalized feedback
        """
        # Combine template-based and LLM-generated feedback
        template_feedback = self._generate_template_feedback(
            comparison_result, 
            problem_analysis, 
            learning_history
        )
        
        # Use LLM to generate more personalized feedback
        llm_feedback = self._generate_llm_feedback(
            comparison_result, 
            problem_analysis, 
            learning_history
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
        problem_analysis: Dict[str, Any],
        learning_history: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """
        Generate feedback using templates.
        
        Args:
            comparison_result: Result of comparing student and expert reviews
            problem_analysis: Analysis of the problems in the code
            learning_history: Optional history of student learning progress
            
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
        improvement_areas = self._identify_improvement_areas(comparison_result, learning_history)
        
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
        problem_analysis: Dict[str, Any],
        learning_history: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate personalized feedback using the LLM.
        
        Args:
            comparison_result: Result of comparing student and expert reviews
            problem_analysis: Analysis of the problems in the code
            learning_history: Optional history of student learning progress
            
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
        if learning_history:
            # Extract relevant information from learning history
            strengths = learning_history.get("strengths", [])
            weaknesses = learning_history.get("weaknesses", [])
            completed_exercises = learning_history.get("completed_exercises", 0)
            
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
    
    def _identify_improvement_areas(
        self, 
        comparison_result: Dict[str, Any],
        learning_history: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """
        Identify areas where the student can improve.
        
        Args:
            comparison_result: Result of comparing student and expert reviews
            learning_history: Optional history of student learning progress
            
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