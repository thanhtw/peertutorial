"""
Review Analytics Agent for Peer Code Review Tutorial System

This agent is responsible for comparing student reviews against expert reviews
and analyzing the quality and completeness of the student's code review.
"""

import json
import re
import requests
import os
from typing import Dict, List, Any, Optional, Tuple, Union
import backoff

class ReviewAnalyticsAgent:
    """
    Review Analytics Agent that analyzes the quality of student reviews.
    
    This agent:
    1. Compares student reviews with expert reviews
    2. Identifies issues caught or missed by the student
    3. Evaluates the quality and thoroughness of the review
    4. Provides detailed feedback on review performance
    """
    
    def __init__(self):
        """Initialize the Review Analytics Agent."""
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