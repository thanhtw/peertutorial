"""
Problem Tracker Agent for Peer Code Review Tutorial System

This agent is responsible for analyzing and tracking the "ground truth" of issues
in the generated code, providing a baseline for evaluating student reviews.
"""

import json
import re
from typing import Dict, List, Any, Optional
import hashlib

class ProblemTrackerAgent:
    """
    Problem Tracker Agent that maintains the ground truth of issues in code snippets.
    
    This agent:
    1. Analyzes compilation errors and checkstyle issues
    2. Identifies and categorizes all issues in the code
    3. Creates a database of known issues for comparison with student reviews
    4. Assigns difficulty and educational value ratings to issues
    """
    
    def __init__(self):
        """Initialize the Problem Tracker Agent."""
        # Initialize issues database
        self.issues_database = {}
    
    def analyze_code_issues(
        self, 
        code_snippet: str, 
        compile_results: Dict[str, Any], 
        checkstyle_results: Dict[str, Any],
        build_errors: Optional[Dict] = None,
        checkstyle_errors: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Analyze a code snippet to identify all issues.
        
        Args:
            code_snippet: The Java code snippet
            compile_results: Results from compiling the code
            checkstyle_results: Results from running checkstyle
            build_errors: Database of known build errors
            checkstyle_errors: Database of known checkstyle errors
            
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
                code_snippet, 
                build_errors
            )
            issues.extend(compile_issues)
        
        # Process checkstyle issues if any
        if checkstyle_results["issues"]:
            style_issues = self._analyze_checkstyle_issues(
                checkstyle_results["issues"], 
                code_snippet, 
                checkstyle_errors
            )
            issues.extend(style_issues)
        
        # Identify additional logical or semantic issues
        logical_issues = self._identify_logical_issues(code_snippet, build_errors)
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
        code_snippet: str,
        build_errors: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze compilation error output to identify specific issues.
        
        Args:
            error_output: Compilation error output
            code_snippet: The Java code snippet
            build_errors: Database of known build errors
            
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
                if build_errors:
                    for category in build_errors:
                        for error in build_errors[category]:
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
        code_snippet: str,
        checkstyle_errors: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze checkstyle issues to categorize and prioritize them.
        
        Args:
            style_issues: List of checkstyle issues
            code_snippet: The Java code snippet
            checkstyle_errors: Database of known checkstyle errors
            
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
            category = self._determine_checkstyle_category(rule, checkstyle_errors)
            severity = self._determine_issue_severity(category, rule)
            
            # Find detailed description from checkstyle_errors if available
            description = message
            if checkstyle_errors:
                for category_name, checks in checkstyle_errors.items():
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
    
    def _determine_checkstyle_category(
        self, 
        rule: str,
        checkstyle_errors: Optional[Dict] = None
    ) -> str:
        """
        Determine the category of a checkstyle rule.
        
        Args:
            rule: The checkstyle rule name
            checkstyle_errors: Database of known checkstyle errors
            
        Returns:
            Category name as a string
        """
        # If we have the checkstyle_errors database, look up the category
        if checkstyle_errors:
            for category, checks in checkstyle_errors.items():
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
    
    def _identify_logical_issues(
        self, 
        code_snippet: str,
        build_errors: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify potential logical issues in the code.
        
        Args:
            code_snippet: The Java code snippet
            build_errors: Database of known build errors
            
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
        
        # Add additional logical issue checks here as needed
        
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
    
    def get_issue_by_id(self, snippet_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the analysis results for a specific code snippet.
        
        Args:
            snippet_id: The unique ID of the code snippet
            
        Returns:
            Dict with analysis results or None if not found
        """
        return self.issues_database.get(snippet_id)
    
    def get_all_issues(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the entire issues database.
        
        Returns:
            Dict with all analysis results
        """
        return self.issues_database