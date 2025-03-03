"""
Main application module for Peer Code Review Tutorial System.
Updated to include a dedicated Ollama LLM setup section.
"""

import streamlit as st
import subprocess
import os
import sys
import platform
import tempfile
import json
import time
from pathlib import Path
import re
from modules.ai_agent import (
    get_code_review_knowledge,
    generate_code_snippet,
    get_ai_review,
    compare_reviews,
    is_ollama_running,
    start_ollama,
    get_available_models,
    set_ollama_model,
    pull_model,
    OLLAMA_MODEL,
    OLLAMA_URL
)
from modules.compiler import compile_java, ensure_java_installed, get_java_version, extract_class_name
from modules.checkstyle import run_checkstyle, check_checkstyle_available
from ollama_setup import _render_llm_setup_section

# Set page configuration
st.set_page_config(
    page_title="Peer Code Review Tutorial",
    page_icon="📝",
    layout="wide"
)

# Ensure required directories exist
def ensure_directories():
    """
    Ensure required directories exist.
    """
    # Create models directory if it doesn't exist
    models_dir = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
    os.makedirs(models_dir, exist_ok=True)
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create .env file if it doesn't exist
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_file):
        with open(env_file, "w") as f:
            f.write("# Models directory\n")
            f.write(f"MODELS_DIR={models_dir}\n")
            f.write("\n# Ollama Settings\n")
            f.write("OLLAMA_URL=http://localhost:11434\n")
            f.write(f"OLLAMA_MODEL={OLLAMA_MODEL}\n")

# Initialize session state variables
def initialize_session_state():
    """Initialize all session state variables with default values."""
    if "step" not in st.session_state:
        st.session_state.step = 1
    if "knowledge" not in st.session_state:
        st.session_state.knowledge = ""
    if "code_snippet" not in st.session_state:
        st.session_state.code_snippet = ""
    if "compile_results" not in st.session_state:
        st.session_state.compile_results = None
    if "checkstyle_results" not in st.session_state:
        st.session_state.checkstyle_results = None
    if "ai_review" not in st.session_state:
        st.session_state.ai_review = ""
    if "student_review" not in st.session_state:
        st.session_state.student_review = ""
    if "comparison" not in st.session_state:
        st.session_state.comparison = None
    # Add model selection state
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = OLLAMA_MODEL
    if "available_models" not in st.session_state:
        st.session_state.available_models = []
    # Add model initialization state 
    if "model_initialized" not in st.session_state:
        st.session_state.model_initialized = False
    if "initialization_message" not in st.session_state:
        st.session_state.initialization_message = ""
    if "system_ready" not in st.session_state:
        st.session_state.system_ready = False
    # Add model parameters 
    if "model_params" not in st.session_state:
        st.session_state.model_params = {
            "temperature": 0.7,
            "max_tokens": 1024
        }

# Verify system requirements
def check_system_requirements():
    """
    Check if system meets requirements to run the application.
    
    Returns:
        tuple: (ready, details) where ready is a boolean and details is a dict
    """
    details = {
        "java_installed": False,
        "java_version": None,
        "ollama_running": False,
        "checkstyle_available": False,
        "messages": []
    }
    
    # Check Java installation
    java_installed = ensure_java_installed()
    details["java_installed"] = java_installed
    details["java_version"] = get_java_version()
    
    if not java_installed:
        details["messages"].append("Java is not installed. Please install Java to compile and analyze Java code.")
    
    # Check Ollama
    ollama_running = is_ollama_running()
    details["ollama_running"] = ollama_running
    
    if not ollama_running:
        details["messages"].append(f"Ollama is not running at {OLLAMA_URL}. The application will attempt to start it.")
    
    # Check Checkstyle
    checkstyle_available, checkstyle_message = check_checkstyle_available()
    details["checkstyle_available"] = checkstyle_available
    
    if not checkstyle_available:
        details["messages"].append(f"Checkstyle is not available: {checkstyle_message}")
    
    # System is ready if Java is installed
    # (We can still function without Ollama running as we'll try to start it)
    # (We will download Checkstyle if it's missing)
    ready = java_installed
    
    return ready, details

# Display system status
def display_system_status():
    """Display system status in the sidebar."""
    # Check requirements
    ready, details = check_system_requirements()
    
    st.sidebar.header("System Status")
    
    # Java status
    if details["java_installed"]:
        st.sidebar.success(f"✅ Java is installed: {details['java_version']}")
    else:
        st.sidebar.error("❌ Java is not installed")
        st.sidebar.info("Please install Java Development Kit (JDK) to use this application.")
    
    # Ollama status
    if details["ollama_running"]:
        st.sidebar.success(f"✅ Ollama is running at {OLLAMA_URL}")
    else:
        st.sidebar.warning(f"⚠️ Ollama is not running at {OLLAMA_URL}")
        if st.sidebar.button("Start Ollama"):
            status_placeholder = st.sidebar.empty()
            status_placeholder.info("Starting Ollama...")
            if start_ollama():
                status_placeholder.success("✅ Ollama started successfully!")
                details["ollama_running"] = True
                st.rerun()
            else:
                status_placeholder.error("❌ Failed to start Ollama")
                st.sidebar.info("Please start Ollama manually with 'ollama serve'")
    
    # Checkstyle status
    if details["checkstyle_available"]:
        st.sidebar.success("✅ Checkstyle is available")
    else:
        st.sidebar.warning("⚠️ Checkstyle will be downloaded when needed")
    
    # Overall status
    if ready:
        st.session_state.system_ready = True
        return True
    else:
        st.session_state.system_ready = False
        st.warning("⚠️ System is not fully ready. Please address the issues above.")
        return False

# Main function
def main():
    """
    Main application function.
    
    This function initializes the application, handles session state,
    and renders the appropriate content based on the current step.
    """
    # Ensure required directories exist
    ensure_directories()
    
    # Initialize session state
    initialize_session_state()
    
    # Page header
    st.title("🧠 Peer Code Review Tutorial System")
    
    # Sidebar content
    st.sidebar.title("Peer Code Review")
    
    # Check and display system status
    system_ready = display_system_status()
    
    # If system is not ready, don't proceed further
    if not system_ready and not st.session_state.system_ready:
        st.info("Please address the system requirements above to continue.")
        return
    
    # Organize sections with proper numbering for main content
    section_num = 1
    
    # LLM Model Setup Section
    _render_llm_setup_section(section_num)
    section_num += 1
    
    # Handle different steps of the tutorial
    if not st.session_state.model_initialized:
        st.warning("Please initialize an Ollama model before continuing.")
    elif st.session_state.step == 1:
        # Step 1: Introduction and knowledge about peer code review
        render_introduction_step(section_num)
    elif st.session_state.step == 2:
        # Step 2: Code snippet review
        render_code_review_step(section_num)
    elif st.session_state.step == 3:
        # Step 3: Comparison and feedback
        render_comparison_step(section_num)

# Move to the next step in the tutorial
def next_step():
    """Move to the next step in the tutorial."""
    st.session_state.step += 1

# Reset the tutorial to the beginning state
def restart_tutorial():
    """Reset the tutorial to the beginning state."""
    st.session_state.step = 1
    st.session_state.code_snippet = ""
    st.session_state.compile_results = None
    st.session_state.checkstyle_results = None
    st.session_state.ai_review = ""
    st.session_state.student_review = ""
    st.session_state.comparison = None

# Generate a new code snippet for review
def generate_new_snippet():
    """Generate a new code snippet and analyze it."""
    # First ensure a model is initialized
    if not st.session_state.model_initialized:
        st.error("No model is initialized. Please initialize a model first.")
        return
    
    with st.spinner("Generating code snippet and analyzing..."):
        try:
            # Generate code snippet using AI
            code_snippet = generate_code_snippet()
            
            # Check if we got a valid response or an error message
            if "Ollama is not running" in code_snippet or "Error communicating" in code_snippet:
                st.error(code_snippet)
                return
                
            st.session_state.code_snippet = code_snippet
            
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(suffix=".java", delete=False) as tmp_file:
                tmp_file.write(code_snippet.encode())
                tmp_path = tmp_file.name
            
            # Compile the Java code
            compile_results = compile_java(tmp_path)
            st.session_state.compile_results = compile_results
            
            # Run checkstyle on the Java code
            checkstyle_results = run_checkstyle(tmp_path)
            st.session_state.checkstyle_results = checkstyle_results
            
            # Generate AI review based on compilation and checkstyle results
            ai_review = get_ai_review(code_snippet, compile_results, checkstyle_results)
            
            # Check if we got a valid response or an error message
            if "Ollama is not running" in ai_review or "Error communicating" in ai_review:
                st.error(ai_review)
                return
                
            st.session_state.ai_review = ai_review
            
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            next_step()
        except Exception as e:
            st.error(f"Error generating code snippet: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")

# Submit student review and compare with AI review
def submit_review():
    """Submit student review and generate comparison."""
    with st.spinner("Analyzing your review..."):
        try:
            # Get comparison between AI and student reviews
            comparison = compare_reviews(
                st.session_state.ai_review,
                st.session_state.student_review
            )
            
            # Check if we got a valid response or an error message
            if isinstance(comparison, str) and ("Ollama is not running" in comparison or "Error communicating" in comparison):
                st.error(comparison)
                return
                
            st.session_state.comparison = comparison
            next_step()
        except Exception as e:
            st.error(f"Error analyzing review: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")

# Render the introduction step
def render_introduction_step(section_num):
    """
    Render the introduction and knowledge about peer code review.
    
    Args:
        section_num (int): Section number for header
    """
    st.header(f"{section_num}. Welcome to the Peer Code Review Tutorial")
    
    # Load knowledge about peer code review if not already loaded
    if not st.session_state.knowledge:
        with st.spinner("Loading code review knowledge..."):
            if is_ollama_running() and st.session_state.model_initialized:
                knowledge = get_code_review_knowledge()
                st.session_state.knowledge = knowledge
            else:
                st.session_state.knowledge = """
                ## Peer Code Review
                
                Peer code review is a software quality assurance practice where developers review each other's code
                to identify bugs, improve code quality, and ensure adherence to standards.
                
                ### Key Benefits:
                - Identifies bugs and issues early
                - Ensures code quality and consistency
                - Facilitates knowledge sharing
                - Improves overall software quality
                
                ### Best Practices:
                1. Be constructive, not critical
                2. Focus on code, not the coder
                3. Review code in small chunks
                4. Provide specific, actionable feedback
                5. Use a checklist for consistency
                
                Please initialize an Ollama model to get more detailed information.
                """
    
    st.markdown(st.session_state.knowledge)
    
    st.subheader("How this tutorial works:")
    st.write("""
    1. You'll be shown a Java code snippet with potential issues
    2. The system will compile the code and run style checks
    3. Review the code and write your comments
    4. Compare your review with an AI-generated review
    5. Receive feedback on your reviewing skills
    """)
    
    # Only enable the button if a model is initialized
    if st.session_state.model_initialized:
        st.button("Start Practice", on_click=generate_new_snippet)
    else:
        st.button("Start Practice", disabled=True, help="Please initialize a model first")
        st.warning("Please initialize an Ollama model using the section above.")

# Render the code review step
def render_code_review_step(section_num):
    """
    Render the code snippet review step.
    
    Args:
        section_num (int): Section number for header
    """
    st.header(f"{section_num}. Review This Code Snippet")
    
    # Display code snippet
    st.code(st.session_state.code_snippet, language="java")
    
    # Display compilation results
    st.subheader("Compilation Results")
    if st.session_state.compile_results["success"]:
        st.success("✅ Compilation successful")
    else:
        st.error("❌ Compilation failed:")
        st.code(st.session_state.compile_results["errors"])
    
    # Display checkstyle results
    st.subheader("Code Quality Results")
    if not st.session_state.checkstyle_results["issues"]:
        st.success("✅ No style issues found")
    else:
        st.warning(f"⚠️ {len(st.session_state.checkstyle_results['issues'])} style issues found:")
        
        # Create a dataframe to display issues nicely
        import pandas as pd
        issues_df = pd.DataFrame(st.session_state.checkstyle_results["issues"])
        
        # Select and rename columns for display
        if not issues_df.empty:
            # Make sure all expected columns exist, add empty ones if needed
            for col in ["line", "rule", "message", "severity"]:
                if col not in issues_df.columns:
                    issues_df[col] = ""
            
            display_df = issues_df[["line", "rule", "message", "severity"]]
            display_df.columns = ["Line", "Rule", "Message", "Severity"]
            display_df = display_df.sort_values("Line")
            st.dataframe(display_df)
    
    # AI review (collapsible)
    with st.expander("Show AI Review"):
        st.markdown(st.session_state.ai_review)
    
    # Student review input
    st.subheader("Submit Your Review")
    st.session_state.student_review = st.text_area(
        "Your Code Review Comments:",
        height=200,
        placeholder="Enter your code review comments here. Consider code quality, potential bugs, and improvements."
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        st.button("Back", on_click=restart_tutorial)
    with col2:
        submit_button = st.button("Submit Review")
        if submit_button and st.session_state.student_review.strip():
            submit_review()

# Render the comparison step
def render_comparison_step(section_num):
    """
    Render the comparison and feedback step.
    
    Args:
        section_num (int): Section number for header
    """
    st.header(f"{section_num}. Review Comparison")
    
    # Display student review and AI review side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Your Review")
        st.markdown(st.session_state.student_review)
    
    with col2:
        st.subheader("AI Review")
        st.markdown(st.session_state.ai_review)
    
    # Display feedback
    st.subheader("Feedback on Your Review")
    st.markdown(st.session_state.comparison["feedback"])
    
    # Display strengths and areas for improvement
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Strengths of Your Review")
        for strength in st.session_state.comparison["strengths"]:
            st.markdown(f"- {strength}")
    
    with col2:
        st.subheader("Areas for Improvement")
        for improvement in st.session_state.comparison["improvements"]:
            st.markdown(f"- {improvement}")
    
    # Display missed issues
    if st.session_state.comparison["missed_issues"]:
        st.subheader("Issues You Might Have Missed")
        for issue in st.session_state.comparison["missed_issues"]:
            st.markdown(f"- {issue}")
    
    # Navigation buttons
    col1, col2 = st.columns([1, 5])
    with col1:
        st.button("Try Another Example", on_click=generate_new_snippet)
    with col2:
        st.button("Restart Tutorial", on_click=restart_tutorial)

# Run the application
if __name__ == "__main__":
    main()