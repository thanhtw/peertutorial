"""
Orchestrator Agent for Peer Code Review Tutorial System

This agent coordinates all other components of the system, managing the flow of
information and ensuring each specialized agent fulfills its role correctly.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import streamlit as st

# Import specialized agents
from agents.code_generator_agent import CodeGeneratorAgent
from agents.problem_tracker_agent import ProblemTrackerAgent
from agents.review_analytics_agent import ReviewAnalyticsAgent
from agents.feedback_agent import FeedbackAgent
from agents.progress_tracker_agent import ProgressTrackerAgent

# Import utilities
from modules.compiler import compile_java, extract_class_name
from modules.checkstyle import run_checkstyle
from utils.ollama_utils import is_ollama_running, start_local_ollama, DEFAULT_OLLAMA_MODEL

class OrchestratorAgent:
    """
    Orchestrator Agent that coordinates all other agents in the system.
    
    This agent:
    1. Maintains session state
    2. Initializes and coordinates other agents
    3. Manages the overall learning workflow
    4. Provides communication between components
    """
    
    def __init__(self):
        """Initialize the Orchestrator Agent and all specialized agents."""
        # Initialize specialized agents
        self.code_generator = CodeGeneratorAgent()
        self.problem_tracker = ProblemTrackerAgent()
        self.review_analytics = ReviewAnalyticsAgent()
        self.feedback_agent = FeedbackAgent()
        self.progress_tracker = ProgressTrackerAgent()
        
        # System state
        self.model_initialized = False
        self.selected_model = DEFAULT_OLLAMA_MODEL
        
        # Load error databases
        self._load_error_databases()
    
    def _load_error_databases(self):
        """Load build errors and checkstyle errors from JSON files."""
        try:
            # Get the base directory
            base_dir = Path(__file__).parent.parent
            
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
            return False
    
    def initialize_model(self, model_name: str, temperature: float = 0.7, max_tokens: int = 1024) -> bool:
        """
        Initialize the LLM model for all agents.
        
        Args:
            model_name: Name of the model to use
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not is_ollama_running():
            if not start_local_ollama():
                return False
        
        # Initialize model for each agent that requires it
        try:
            self.code_generator.set_model(model_name, temperature, max_tokens)
            self.review_analytics.set_model(model_name, temperature, max_tokens)
            self.feedback_agent.set_model(model_name, temperature, max_tokens)
            
            self.model_initialized = True
            self.selected_model = model_name
            return True
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            return False
    
    def generate_exercise(self, difficulty: str = "medium", focus_areas: List[str] = None) -> Dict[str, Any]:
        """
        Generate a new code review exercise.
        
        Args:
            difficulty: Difficulty level (easy, medium, hard)
            focus_areas: List of areas to focus on (e.g., ["naming", "exceptions", "performance"])
            
        Returns:
            Dict with exercise data
        """
        if not self.model_initialized:
            return {"success": False, "error": "Model not initialized"}
        
        # Use Code Generator Agent to create code with intentional issues
        code_snippet = self.code_generator.generate_code_snippet(
            difficulty=difficulty,
            focus_areas=focus_areas,
            build_errors=self.build_errors,
            checkstyle_errors=self.checkstyle_errors
        )
        
        # Compile the generated code to verify issues
        compile_results = self._compile_code(code_snippet)
        
        # Run checkstyle on the code
        checkstyle_results = self._run_checkstyle(code_snippet)
        
        # Use Problem Tracker to analyze and store the ground truth of issues
        problem_analysis = self.problem_tracker.analyze_code_issues(
            code_snippet, 
            compile_results, 
            checkstyle_results,
            build_errors=self.build_errors,
            checkstyle_errors=self.checkstyle_errors
        )
        
        # Generate expert review for later comparison
        expert_review = self.code_generator.generate_expert_review(
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
            "focus_areas": focus_areas
        }
    
    def _compile_code(self, code_snippet: str) -> Dict[str, Any]:
        """
        Compile the Java code snippet.
        
        Args:
            code_snippet: Java code to compile
            
        Returns:
            Dict with compilation results
        """
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".java", delete=False) as tmp_file:
            tmp_file.write(code_snippet.encode())
            tmp_path = tmp_file.name
        
        try:
            # Compile the Java code
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
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".java", delete=False) as tmp_file:
            tmp_file.write(code_snippet.encode())
            tmp_path = tmp_file.name
        
        try:
            # Run checkstyle
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
        
        # Use Review Analytics Agent to compare reviews
        comparison = self.review_analytics.compare_reviews(
            exercise_data["expert_review"],
            student_review,
            exercise_data["problem_analysis"]
        )
        
        # Track student progress
        self.progress_tracker.update_student_progress(
            comparison,
            exercise_data["difficulty"],
            exercise_data["focus_areas"]
        )
        
        # Generate personalized feedback based on comparison
        feedback = self.feedback_agent.generate_feedback(
            comparison,
            exercise_data["problem_analysis"],
            self.progress_tracker.get_learning_history()
        )
        
        return {
            "success": True,
            "comparison": comparison,
            "feedback": feedback,
            "progress": self.progress_tracker.get_current_progress()
        }
    
    def recommend_next_exercise(self) -> Dict[str, Any]:
        """
        Recommend the next exercise based on student progress.
        
        Returns:
            Dict with recommendation data
        """
        # Use Progress Tracker to recommend next exercise parameters
        recommendation = self.progress_tracker.recommend_next_exercise()
        
        return {
            "difficulty": recommendation["difficulty"],
            "focus_areas": recommendation["focus_areas"],
            "rationale": recommendation["rationale"]
        }
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the student's learning progress.
        
        Returns:
            Dict with learning summary data
        """
        return self.progress_tracker.generate_learning_summary()