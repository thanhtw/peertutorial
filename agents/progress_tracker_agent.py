"""
Progress Tracker Agent for Peer Code Review Tutorial System

This agent is responsible for tracking student learning progress, analyzing
performance trends, and recommending appropriate next exercises.
"""

import json
import time
import os
from typing import Dict, List, Any, Optional
from collections import defaultdict

class ProgressTrackerAgent:
    """
    Progress Tracker Agent that monitors student learning progress.
    
    This agent:
    1. Tracks student performance across exercises
    2. Identifies strengths and weaknesses
    3. Recommends appropriate difficulty and focus areas for next exercises
    4. Generates learning progress summaries
    """
    
    def __init__(self):
        """Initialize the Progress Tracker Agent."""
        self.history = {
            "exercises": [],
            "performance": {},
            "strengths": [],
            "weaknesses": [],
            "completed_exercises": 0,
            "current_level": "beginner",
            "progress_by_category": defaultdict(list),
            "last_updated": time.time()
        }
    
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
        self.history["exercises"].append(exercise)
        self.history["completed_exercises"] += 1
        self.history["last_updated"] = time.time()
        
        # Update performance metrics
        self._update_performance_metrics(exercise)
        
        # Update strengths and weaknesses
        self._update_strengths_and_weaknesses()
        
        # Update student level
        self._update_student_level()
        
        # Save progress (in a real system, this might save to a database)
        self._save_progress()
    
    def _update_performance_metrics(self, exercise: Dict[str, Any]) -> None:
        """
        Update performance metrics based on exercise results.
        
        Args:
            exercise: The completed exercise data
        """
        # Track performance by difficulty
        difficulty = exercise["difficulty"]
        if difficulty not in self.history["performance"]:
            self.history["performance"][difficulty] = {
                "exercises": 0,
                "average_score": 0,
                "total_caught": 0,
                "total_missed": 0
            }
        
        # Update difficulty stats
        performance = self.history["performance"][difficulty]
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
            if category not in self.history["progress_by_category"]:
                self.history["progress_by_category"][category] = []
            
            self.history["progress_by_category"][category].append({
                "timestamp": exercise["timestamp"],
                "caught": count,
                "missed": exercise["missed_by_type"].get(issue_type, 0)
            })
    
    def _update_strengths_and_weaknesses(self) -> None:
        """Update student strengths and weaknesses based on accumulated data."""
        # Reset strengths and weaknesses
        self.history["strengths"] = []
        self.history["weaknesses"] = []
        
        # Look at the last 3 exercises for recent trends
        recent_exercises = self.history["exercises"][-3:]
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
                    self.history["strengths"].append(issue_type)
                elif catch_rate <= 0.4 and total >= 3:
                    self.history["weaknesses"].append(issue_type)
        
        # Add difficulty-based strengths/weaknesses
        for difficulty, performance in self.history["performance"].items():
            if performance["exercises"] >= 2:
                if performance["average_score"] >= 80:
                    self.history["strengths"].append(f"{difficulty} difficulty exercises")
                elif performance["average_score"] <= 50:
                    self.history["weaknesses"].append(f"{difficulty} difficulty exercises")
    
    def _update_student_level(self) -> None:
        """Update the student's current level based on performance."""
        # Get performance at each difficulty level
        easy_perf = self.history["performance"].get("easy", {"average_score": 0, "exercises": 0})
        medium_perf = self.history["performance"].get("medium", {"average_score": 0, "exercises": 0})
        hard_perf = self.history["performance"].get("hard", {"average_score": 0, "exercises": 0})
        
        # Determine level based on performance
        if hard_perf["exercises"] >= 3 and hard_perf["average_score"] >= 70:
            self.history["current_level"] = "expert"
        elif medium_perf["exercises"] >= 3 and medium_perf["average_score"] >= 70:
            self.history["current_level"] = "advanced"
        elif easy_perf["exercises"] >= 3 and easy_perf["average_score"] >= 70:
            self.history["current_level"] = "intermediate"
        else:
            self.history["current_level"] = "beginner"
    
    def _save_progress(self) -> None:
        """Save progress data to storage."""
        # In a real system, this would save to a database or file
        # For this example, we'll just print a summary
        print(f"Progress updated: Level={self.history['current_level']}, "
              f"Completed={self.history['completed_exercises']}, "
              f"Strengths={len(self.history['strengths'])}, "
              f"Weaknesses={len(self.history['weaknesses'])}")
        
        # For a more permanent solution, could save to a file:
        # with open('progress.json', 'w') as f:
        #     json.dump(self.history, f, indent=2)
    
    def get_learning_history(self) -> Dict[str, Any]:
        """
        Get the student's learning history.
        
        Returns:
            Dict with learning history
        """
        return self.history
    
    def get_current_progress(self) -> Dict[str, Any]:
        """
        Get the student's current progress summary.
        
        Returns:
            Dict with progress summary
        """
        # Calculate recent trend (last 3 exercises)
        recent_scores = [ex["overall_score"] for ex in self.history["exercises"][-3:]]
        
        trend = 0  # neutral
        if len(recent_scores) >= 2:
            trend = 1 if recent_scores[-1] > recent_scores[0] else (-1 if recent_scores[-1] < recent_scores[0] else 0)
        
        return {
            "level": self.history["current_level"],
            "completed_exercises": self.history["completed_exercises"],
            "strengths": self.history["strengths"],
            "weaknesses": self.history["weaknesses"],
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
        if self.history["completed_exercises"] == 0:
            return {
                "difficulty": "easy",
                "focus_areas": ["naming", "structure"],
                "rationale": "Starting with the basics of code review"
            }
        
        # Decide on difficulty
        current_level = self.history["current_level"]
        difficulty_mapping = {
            "beginner": "easy",
            "intermediate": "medium",
            "advanced": "medium",  # occasionally give hard exercises
            "expert": "hard"
        }
        
        recommended_difficulty = difficulty_mapping.get(current_level, "easy")
        
        # Occasionally increase difficulty to challenge the student
        # or decrease it if they're struggling
        last_exercise = self.history["exercises"][-1]
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
        for weakness in self.history["weaknesses"][:2]:  # Limit to 2 weaknesses
            for issue_type, focus_areas in issue_to_focus.items():
                if issue_type in weakness.lower():
                    recommended_focus.extend(focus_areas)
        
        # If no specific weaknesses, provide balanced focus
        if not recommended_focus:
            # Include at least one strength area to maintain confidence
            if self.history["strengths"]:
                strength_area = self.history["strengths"][0]
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
        if self.history["weaknesses"]:
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
        if not self.history["exercises"]:
            return {
                "status": "No exercises completed yet",
                "recommendations": "Start with basic code review exercises"
            }
        
        # Calculate overall statistics
        total_caught = sum(ex["caught_issues_count"] for ex in self.history["exercises"])
        total_missed = sum(ex["missed_issues_count"] for ex in self.history["exercises"])
        total_issues = total_caught + total_missed
        
        catch_rate = total_caught / total_issues if total_issues > 0 else 0
        average_score = sum(ex["overall_score"] for ex in self.history["exercises"]) / len(self.history["exercises"])
        
        # Calculate progress over time
        scores_over_time = [
            {"timestamp": ex["timestamp"], "score": ex["overall_score"]}
            for ex in self.history["exercises"]
        ]
        
        # Generate text summary
        if self.history["completed_exercises"] <= 3:
            progress_summary = "You're just getting started with code reviews. Keep practicing to develop your skills."
        else:
            progress_summary = f"You've completed {self.history['completed_exercises']} exercises and are at the {self.history['current_level']} level."
            
            # Add trend information
            recent_scores = [ex["overall_score"] for ex in self.history["exercises"][-3:]]
            if len(recent_scores) >= 3:
                if recent_scores[2] > recent_scores[0]:
                    progress_summary += " Your scores are improving!"
                elif recent_scores[2] < recent_scores[0]:
                    progress_summary += " Your recent scores have decreased slightly."
                else:
                    progress_summary += " Your scores have been consistent recently."
        
        # Generate recommendations
        if self.history["weaknesses"]:
            recommendations = f"Focus on improving your {', '.join(self.history['weaknesses'])}."
        else:
            recommendations = "Continue to practice a variety of code review exercises to maintain your skills."
        
        # Add next steps based on level
        if self.history["current_level"] == "beginner":
            next_steps = "Practice identifying common Java errors and style issues."
        elif self.history["current_level"] == "intermediate":
            next_steps = "Work on identifying subtle logical errors and performance issues."
        elif self.history["current_level"] == "advanced":
            next_steps = "Challenge yourself with complex code reviews and focus on security concerns."
        else:  # expert
            next_steps = "Consider reviewing real-world code in open source projects to further refine your skills."
        
        return {
            "status": progress_summary,
            "level": self.history["current_level"],
            "completed_exercises": self.history["completed_exercises"],
            "strengths": self.history["strengths"],
            "weaknesses": self.history["weaknesses"],
            "average_score": round(average_score, 1),
            "catch_rate": round(catch_rate * 100, 1),
            "scores_over_time": scores_over_time,
            "recommendations": recommendations,
            "next_steps": next_steps
        }