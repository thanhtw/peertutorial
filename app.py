"""
Main application module for Peer Code Review Tutorial System.
This app provides an interactive environment for learning code review practices.
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
import requests
from modules.ai_agent import (
    get_code_review_knowledge,
    generate_code_snippet,
    get_ai_review,
    compare_reviews,
    OLLAMA_MODEL,
    OLLAMA_URL,
    is_ollama_running
)
from modules.compiler import compile_java
from modules.checkstyle import run_checkstyle

# Try to import the model utilities
try:
    from utils.ollama_model_check import (
        check_ollama_server,
        get_pulled_models,
        initialize_best_model,
        check_and_pull_model,
        get_model_details,
        find_suitable_model
    )
    HAVE_MODEL_UTILS = True
except ImportError:
    HAVE_MODEL_UTILS = False

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
    os.makedirs(os.path.join(models_dir, "ollama"), exist_ok=True)
    
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
    if "model_details" not in st.session_state:
        st.session_state.model_details = {}

# Check and initialize Ollama models at startup
def initialize_ollama_models():
    """
    Check for available Ollama models and initialize the best one.
    
    Returns:
        bool: True if a model was initialized, False otherwise
    """
    # Set up the initialization message placeholder
    init_message = st.empty()
    init_message.info("Initializing Ollama models...")
    
    # Check if Ollama server is running
    if not is_ollama_running():
        init_message.warning("⚠️ Ollama server is not running. Start it to use models.")
        return False
    
    # Check if we have the model utilities
    if HAVE_MODEL_UTILS:
        # Use the model utilities for better initialization
        selected_model, all_models = find_suitable_model()
        
        if selected_model:
            # Store models in session state
            model_names = [model["name"] for model in all_models]
            st.session_state.available_models = all_models
            
            # Get model details
            model_details = get_model_details(selected_model)
            st.session_state.model_details = model_details
            
            # Set the selected model
            st.session_state.selected_model = selected_model
            st.session_state.model_initialized = True
            
            # Update initialization message
            size_info = next((m.get("size", "") for m in all_models if m["name"] == selected_model), "")
            init_message.success(f"✅ Initialized model: {selected_model} ({size_info})")
            
            st.session_state.initialization_message = (
                f"Using {selected_model} ({size_info}) out of {len(all_models)} available models."
            )
            
            return True
        else:
            # No models found, try to pull the default model
            init_message.warning(f"No models found. Attempting to pull {OLLAMA_MODEL}...")
            if check_and_pull_model(OLLAMA_MODEL):
                init_message.info(f"Started pulling {OLLAMA_MODEL}. Please wait for the download to complete.")
                st.session_state.initialization_message = (
                    f"Please wait while {OLLAMA_MODEL} is being pulled. "
                    "Refresh the page when download is complete."
                )
            else:
                init_message.error(f"Failed to pull {OLLAMA_MODEL}. Please pull a model manually.")
                st.session_state.initialization_message = (
                    "Failed to pull model. Please pull a model manually using "
                    f"`ollama pull {OLLAMA_MODEL}` in a terminal."
                )
            return False
    else:
        # Fallback to basic initialization 
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                
                if models:
                    # Store models in session state
                    st.session_state.available_models = models
                    model_names = [model["name"] for model in models]
                    
                    # Select a model
                    if OLLAMA_MODEL in model_names:
                        selected_model = OLLAMA_MODEL
                    else:
                        selected_model = model_names[0]
                    
                    # Set the selected model
                    st.session_state.selected_model = selected_model
                    st.session_state.model_initialized = True
                    
                    # Update initialization message
                    init_message.success(f"✅ Initialized model: {selected_model}")
                    st.session_state.initialization_message = (
                        f"Using {selected_model} out of {len(models)} available models."
                    )
                    
                    return True
                else:
                    # No models found
                    init_message.warning(f"No models found. Please pull {OLLAMA_MODEL} manually.")
                    st.session_state.initialization_message = (
                        f"No models found. Please pull {OLLAMA_MODEL} manually."
                    )
                    return False
            else:
                init_message.error("Failed to get models from Ollama.")
                return False
        except Exception as e:
            init_message.error(f"Error initializing models: {str(e)}")
            return False

# Display selected model information
def display_model_info():
    """Display information about the selected model."""
    selected_model = st.session_state.get("selected_model", OLLAMA_MODEL)
    
    # Create a section for model information
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Information")
    
    # Show selected model
    st.sidebar.success(f"✅ Using model: **{selected_model}**")
    
    # If we have model details, show them
    model_details = st.session_state.get("model_details", {})
    if model_details:
        with st.sidebar.expander("Model Details"):
            # Show model parameters
            if "parameters" in model_details:
                st.write(f"Parameters: {model_details['parameters']}")
            
            # Show model family
            if "family" in model_details:
                st.write(f"Family: {model_details['family']}")
                
            # Show model description
            if "description" in model_details:
                st.write(f"Description: {model_details['description']}")
                
            # Show license
            if "license" in model_details:
                st.write(f"License: {model_details['license']}")
    
    # Show initialization message if any
    init_message = st.session_state.get("initialization_message", "")
    if init_message:
        st.sidebar.info(init_message)

# Render the model selector widget with all pulled models
def render_model_selector():
    """Render a dropdown to select from available models."""
    # Get available models
    available_models = st.session_state.get("available_models", [])
    
    if not available_models:
        st.sidebar.warning("No models available. Please pull some models first.")
        return
    
    # Create a dictionary for the selectbox
    model_options = {}
    for model in available_models:
        # Format model description with size if available
        model_name = model["name"]
        model_size = model.get("size", "")
        if model_size:
            display_name = f"{model_name} ({model_size})"
        else:
            display_name = model_name
        model_options[display_name] = model_name
    
    # Default selection - try to find our current selected model
    selected_model = st.session_state.get("selected_model", OLLAMA_MODEL)
    default_index = 0
    model_displays = list(model_options.keys())
    
    for i, display_name in enumerate(model_displays):
        if model_options[display_name] == selected_model:
            default_index = i
            break
    
    # Create the selectbox
    selected_display = st.sidebar.selectbox(
        "Select Model:",
        options=model_displays,
        index=default_index,
        key="model_selectbox"
    )
    
    # Update the selected model if changed
    new_selected_model = model_options[selected_display]
    if new_selected_model != st.session_state.selected_model:
        st.session_state.selected_model = new_selected_model
        
        # Try to get model details if we have model utilities
        if HAVE_MODEL_UTILS:
            st.session_state.model_details = get_model_details(new_selected_model)
        
        # Rerun to update UI with new model
        st.rerun()

# Add model upload section
def render_model_upload():
    """Render a section to upload/pull additional models."""
    # Create a section for adding more models
    with st.sidebar.expander("Add More Models"):
        st.write("### Pull Additional Models")
        
        # List of recommended models to pull
        recommended_models = {
            "llama3:1b": "Llama 3 (1B) - Fastest",
            "llama3:8b": "Llama 3 (8B) - Good balance",
            "deepseek-coder:6.7b": "DeepSeek Coder - Best for code",
            "phi3:mini": "Phi-3 Mini - Small & efficient",
            "gemma:2b": "Gemma 2B - Compact"
        }
        
        # Get already pulled models
        available_models = st.session_state.get("available_models", [])
        pulled_model_names = [model["name"] for model in available_models]
        
        # Filter out already pulled models
        available_to_pull = {k: v for k, v in recommended_models.items() 
                            if k not in pulled_model_names}
        
        if available_to_pull:
            new_model = st.selectbox(
                "Select model to pull:", 
                options=list(available_to_pull.keys()),
                format_func=lambda x: available_to_pull[x],
                key="new_model_pulldown"
            )
            
            if st.button("Pull Selected Model", key="pull_model_button"):
                with st.spinner(f"Pulling {new_model}..."):
                    # Pull the model
                    if HAVE_MODEL_UTILS:
                        success = check_and_pull_model(new_model)
                    else:
                        # Direct API call
                        try:
                            response = requests.post(
                                f"{OLLAMA_URL}/api/pull",
                                json={"name": new_model}
                            )
                            success = response.status_code == 200
                        except Exception:
                            success = False
                    
                    if success:
                        st.success(f"Started pulling {new_model}!")
                        st.info("Please refresh the page when download completes.")
                    else:
                        st.error(f"Failed to pull {new_model}")
        else:
            st.info("All recommended models already pulled!")
            
        # Custom model pull
        st.write("### Pull Custom Model")
        custom_model = st.text_input("Model name:", placeholder="e.g., mistral:7b")
        if custom_model and st.button("Pull Custom Model", key="pull_custom_button"):
            with st.spinner(f"Pulling {custom_model}..."):
                try:
                    response = requests.post(
                        f"{OLLAMA_URL}/api/pull",
                        json={"name": custom_model}
                    )
                    if response.status_code == 200:
                        st.success(f"Started pulling {custom_model}!")
                        st.info("Please refresh the page when complete.")
                    else:
                        st.error(f"Failed to pull {custom_model}")
                except Exception as e:
                    st.error(f"Error pulling model: {str(e)}")

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
    
    # Check and initialize Ollama models
    if not st.session_state.model_initialized:
        initialize_ollama_models()
    
    # Page header
    st.title("🧠 Peer Code Review Tutorial System")
    
    # Sidebar content
    st.sidebar.title("Peer Code Review")
    
    # Display model information
    display_model_info()
    
    # Render model selector if we have models
    if st.session_state.available_models:
        render_model_selector()
        
        # Render model upload section
        render_model_upload()
    
    # Handle different steps of the tutorial
    if st.session_state.step == 1:
        # Step 1: Introduction and knowledge about peer code review
        render_introduction_step()
    elif st.session_state.step == 2:
        # Step 2: Code snippet review
        render_code_review_step()
    elif st.session_state.step == 3:
        # Step 3: Comparison and feedback
        render_comparison_step()

# Render the introduction step
def render_introduction_step():
    """Render the introduction and knowledge about peer code review."""
    st.header("Welcome to the Peer Code Review Tutorial")
    
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
                
                Please wait for Ollama initialization to get more detailed information.
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
        st.button("Start Practice", disabled=True, help="Please wait for model initialization")
        st.warning("Please wait for model initialization or pull a model using the sidebar.")

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
        st.error("No model is initialized. Please wait for initialization or pull a model.")
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
            os.unlink(tmp_path)
            
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

# Render the code review step
def render_code_review_step():
    """Render the code snippet review step."""
    st.header("Review This Code Snippet")
    
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
        st.warning("⚠️ Style issues found:")
        for issue in st.session_state.checkstyle_results["issues"]:
            st.write(f"- Line {issue['line']}: {issue['message']}")
    
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
def render_comparison_step():
    """Render the comparison and feedback step."""
    st.header("Review Comparison")
    
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