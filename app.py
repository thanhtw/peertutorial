"""
Simplified Application for Peer Code Review Tutorial System
Using the unified AI agent architecture instead of multiple agents
"""

import streamlit as st
import time
import json
import pandas as pd
import os
from pathlib import Path
from unified_peer_review_agent import UnifiedPeerReviewAgent

# Set page configuration
st.set_page_config(
    page_title="AI-Enhanced Peer Code Review Tutorial",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state variables
def initialize_session_state():
    """Initialize all session state variables with default values.111"""
    if "step" not in st.session_state:
        st.session_state.step = 1
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "current_exercise" not in st.session_state:
        st.session_state.current_exercise = None
    if "review_result" not in st.session_state:
        st.session_state.review_result = None
    if "student_review" not in st.session_state:
        st.session_state.student_review = ""
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "llama3:1b"
    if "model_initialized" not in st.session_state:
        st.session_state.model_initialized = False
    if "system_ready" not in st.session_state:
        st.session_state.system_ready = False
    if "model_params" not in st.session_state:
        st.session_state.model_params = {
            "temperature": 0.7,
            "max_tokens": 1024
        }

# Ensure directories and files exist
def ensure_directories():
    """Ensure required directories exist."""
    # Create data directory
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Create .env file if it doesn't exist
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write("# Ollama Settings\n")
            f.write("OLLAMA_URL=http://localhost:11434\n")
            f.write("OLLAMA_MODEL=llama3:1b\n")

# Display system status in the sidebar
def display_system_status():
    """Display system status in the sidebar and initialize the agent if needed."""
    st.sidebar.header("System Status")
    
    # Check if Java is installed
    from modules.compiler import ensure_java_installed, get_java_version
    java_installed = ensure_java_installed()
    java_version = get_java_version()
    
    if java_installed:
        st.sidebar.success(f"✅ Java is installed: {java_version}")
    else:
        st.sidebar.error("❌ Java is not installed")
        st.sidebar.info("Please install Java Development Kit (JDK) to use this application.")
        return False
    
    # Initialize agent if not already done
    if st.session_state.agent is None:
        st.session_state.agent = UnifiedPeerReviewAgent()
    
    # Check if Ollama is running
    if st.session_state.agent.is_ollama_running():
        st.sidebar.success(f"✅ Ollama is running at {st.session_state.agent.ollama_url}")
    else:
        st.sidebar.warning(f"⚠️ Ollama is not running")
        if st.sidebar.button("Start Ollama"):
            status_placeholder = st.sidebar.empty()
            status_placeholder.info("Starting Ollama...")
            if st.session_state.agent.start_ollama():
                status_placeholder.success("✅ Ollama started successfully!")
                st.rerun()
            else:
                status_placeholder.error("❌ Failed to start Ollama")
                st.sidebar.info("Please start Ollama manually with 'ollama serve'")
                return False
    
    # Check if Checkstyle is available
    from modules.checkstyle import check_checkstyle_available
    checkstyle_available, checkstyle_message = check_checkstyle_available()
    if checkstyle_available:
        st.sidebar.success("✅ Checkstyle is available")
    else:
        st.sidebar.warning("⚠️ Checkstyle will be downloaded when needed")
    
    return java_installed

# Render LLM setup section
def render_llm_setup_section(section_num: int):
    """
    Render LLM setup section for Ollama models.
    
    Args:
        section_num (int): Section number for header
    """
    st.header(f"{section_num}. Setup AI Model")
    
    agent = st.session_state.agent
    
    if not agent.is_ollama_running():
        st.error("Ollama server is not running. Please start it from the sidebar.")
        return
    
    # Show available models
    st.write("### Available Ollama Models")
    st.write("Select a model from the dropdown. Models with ✅ are already pulled and ready to use.")

    # Get available models
    from utils.ollama_utils import get_ollama_models
    available_models = get_ollama_models()
    
    if not available_models:
        st.warning("Could not retrieve models from Ollama server. Please check your connection.")
        return
    
    # Model selection dropdown
    model_display_names = {}
    for model in available_models:
        if model.get("pulled", False):
            display_name = f"✅ {model['name']} - {model['description']}"
        else:
            display_name = f"⬇️ {model['name']} - {model['description']}"
        model_display_names[display_name] = model["id"]
    
    selected_display_name = st.selectbox(
        "Select Model:",
        list(model_display_names.keys())
    )
    
    if selected_display_name:
        model_id = model_display_names[selected_display_name]
        selected_model = next((m for m in available_models if m["id"] == model_id), None)
        
        if selected_model:
            # Show model information
            st.write(f"**Selected Model:** {selected_model['name']}")
            is_pulled = selected_model.get("pulled", False)
            
            # Create columns for buttons
            col1, col2 = st.columns([1, 1])
            
            # Pull button (only show if not pulled)
            if not is_pulled:
                with col1:
                    if st.button(f"Pull {selected_model['name']} Model"):
                        with st.spinner(f"Pulling {selected_model['name']}..."):
                            from utils.ollama_utils import pull_ollama_model
                            if pull_ollama_model(model_id):
                                st.success(f"Successfully pulled {selected_model['name']}")
                                st.rerun()
                            else:
                                st.error(f"Failed to pull {selected_model['name']}")
            
            # Advanced options
            with st.expander("Model Parameters", expanded=False):
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    help="Higher values make output more random, lower values more deterministic."
                )
                
                max_tokens = st.slider(
                    "Max Tokens",
                    min_value=100,
                    max_value=4096,
                    value=1024,
                    step=100,
                    help="Maximum token length of the generated responses."
                )
                
                # Store parameters in session state for later use
                st.session_state.model_params = {
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            
            # Initialize button
            init_col = col2 if not is_pulled else col1
            with init_col:
                if st.button("Initialize Selected Model", disabled=not is_pulled):
                    if not is_pulled:
                        st.warning("Please pull the model first before initializing.")
                    else:
                        with st.spinner(f"Initializing {selected_model['name']}..."):
                            # Initialize the model in the agent
                            if agent.initialize_model(
                                model_id, 
                                temperature=temperature, 
                                max_tokens=max_tokens
                            ):
                                st.session_state.model_initialized = True
                                st.session_state.selected_model = model_id
                                st.success(f"Successfully initialized {selected_model['name']}!")
                            else:
                                st.error(f"Failed to initialize {selected_model['name']}")
            
            # Show model status if initialized
            if st.session_state.model_initialized and st.session_state.selected_model == model_id:
                st.success(f"✅ {selected_model['name']} is currently initialized and ready to use")
                
                # Show current parameters
                params = st.session_state.model_params
                if params:
                    st.write("Current parameters:")
                    params_str = ", ".join([f"{k}: {v}" for k, v in params.items()])
                    st.code(params_str)

# Generate a new exercise
def generate_new_exercise(difficulty: str = "medium", focus_areas: list = None):
    """
    Generate a new code review exercise.
    
    Args:
        difficulty: Difficulty level (easy, medium, hard)
        focus_areas: List of areas to focus on
    """
    # First ensure a model is initialized
    if not st.session_state.model_initialized:
        st.error("No model is initialized. Please initialize a model first.")
        return False
    
    agent = st.session_state.agent
    
    with st.spinner("Generating code snippet and analyzing..."):
        try:
            # Generate exercise using the agent
            exercise_data = agent.generate_exercise(difficulty, focus_areas)
            
            if not exercise_data["success"]:
                st.error(exercise_data.get("error", "Failed to generate exercise"))
                return False
            
            # Store the exercise in session state
            st.session_state.current_exercise = exercise_data
            
            # Move to next step
            st.session_state.step = 2
            return True
        except Exception as e:
            st.error(f"Error generating exercise: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")
            return False

# Submit student review
def submit_review():
    """
    Submit the student's review for analysis.
    """
    if not st.session_state.current_exercise:
        st.error("No active exercise. Please generate a new exercise first.")
        return False
    
    if not st.session_state.student_review.strip():
        st.warning("Please enter your review before submitting.")
        return False
    
    agent = st.session_state.agent
    
    with st.spinner("Analyzing your review..."):
        try:
            # Analyze student review
            result = agent.analyze_student_review(
                st.session_state.current_exercise,
                st.session_state.student_review
            )
            
            if not result["success"]:
                st.error(result.get("error", "Failed to analyze review"))
                return False
            
            # Store the analysis result
            st.session_state.review_result = result
            
            # Move to next step
            st.session_state.step = 3
            return True
        except Exception as e:
            st.error(f"Error analyzing review: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")
            return False

# Render introduction step
def render_introduction_step(section_num: int):
    """
    Render the introduction and explanation of the tutorial system.
    
    Args:
        section_num (int): Section number for header
    """
    st.header(f"{section_num}. Welcome to the AI-Enhanced Peer Code Review Tutorial")
    
    # Introduction text
    st.markdown("""
    This tutorial system uses AI to help you improve your code review skills for Java.
    The system will:
    
    1. Generate Java code with intentional issues for you to find
    2. Compile the code and run style checks
    3. Analyze your review against an expert assessment
    4. Provide personalized feedback on your review
    5. Track your progress and adapt to your learning needs
    
    Each exercise is designed to challenge your ability to identify different types of issues,
    from compilation errors to style violations and logical bugs.
    """)
    
    # Difficulty selection
    st.subheader("Choose Your Exercise")
    
    col1, col2 = st.columns(2)
    
    with col1:
        difficulty = st.selectbox(
            "Difficulty Level:",
            ["easy", "medium", "hard"],
            index=0,
            help="Easy exercises have more obvious issues, while hard ones have subtle problems."
        )
    
    with col2:
        focus_options = [
            "naming", "structure", "error_handling", 
            "performance", "security", "logic"
        ]
        
        focus_areas = st.multiselect(
            "Focus Areas (Optional):",
            focus_options,
            default=["naming", "structure"],
            help="Select areas you want to practice. Leave empty for a balanced exercise."
        )
    
    # Start button
    start_col1, start_col2, start_col3 = st.columns([1, 2, 1])
    with start_col2:
        if st.button("Start Exercise", use_container_width=True):
            generate_new_exercise(difficulty, focus_areas)

# Render code review step
def render_code_review_step(section_num: int):
    """
    Render the code review exercise step.
    
    Args:
        section_num (int): Section number for header
    """
    st.header(f"{section_num}. Review This Code")
    
    exercise = st.session_state.current_exercise
    if not exercise:
        st.error("No active exercise. Please generate a new exercise.")
        st.button("Return to Introduction", on_click=lambda: setattr(st.session_state, "step", 1))
        return
    
    # Display exercise information
    st.subheader("Exercise Information")
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.write(f"**Difficulty:** {exercise['difficulty'].capitalize()}")
        if exercise.get('focus_areas'):
            st.write(f"**Focus Areas:** {', '.join(area.capitalize() for area in exercise['focus_areas'])}")
    
    # Display code snippet
    st.subheader("Review this Java Code")
    st.code(exercise["code_snippet"], language="java")
    
    # Display compilation results
    st.subheader("Compilation Results")
    if exercise["compile_results"]["success"]:
        st.success("✅ Compilation successful")
    else:
        st.error("❌ Compilation failed:")
        st.code(exercise["compile_results"]["errors"])
    
    # Display checkstyle results
    st.subheader("Code Quality Results")
    if not exercise["checkstyle_results"]["issues"]:
        st.success("✅ No style issues found")
    else:
        st.warning(f"⚠️ {len(exercise['checkstyle_results']['issues'])} style issues found:")
        
        # Create a dataframe to display issues nicely
        issues_df = pd.DataFrame(exercise["checkstyle_results"]["issues"])
        
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
    
    # Student review input
    st.subheader("Submit Your Review")
    review_instructions = """
    Write your review comments here. Consider:
    
    - Compilation errors and warnings
    - Code style and formatting issues
    - Logical errors or bugs
    - Performance issues
    - Suggestions for improvement
    
    Be specific and explain why each issue is problematic.
    """
    st.session_state.student_review = st.text_area(
        "Your Code Review Comments:",
        value=st.session_state.student_review,
        height=250,
        placeholder="Enter your code review comments here...",
        help=review_instructions
    )
    
    # Navigation buttons
    col1, col2 = st.columns([1, 5])
    with col1:
        st.button("Back", on_click=lambda: setattr(st.session_state, "step", 1))
    with col2:
        submit_button = st.button("Submit Review")
        if submit_button:
            submit_review()

# Render feedback step
def render_feedback_step(section_num: int):
    """
    Render the feedback and results step.
    
    Args:
        section_num (int): Section number for header
    """
    st.header(f"{section_num}. Review Feedback")
    
    result = st.session_state.review_result
    if not result:
        st.error("No review result available. Please complete a review first.")
        st.button("Return to Introduction", on_click=lambda: setattr(st.session_state, "step", 1))
        return
    
    # Display overall score and metrics
    score = result["comparison"]["overall_score"]
    metrics = result["comparison"]["metrics"]
    
    score_col1, score_col2, score_col3 = st.columns(3)
    with score_col1:
        st.metric("Overall Score", f"{score}/100")
    with score_col2:
        st.metric("Issues Found", f"{metrics.get('coverage', 0) * 100:.0f}%")
    with score_col3:
        st.metric("Critical Issues Found", f"{metrics.get('severity_weighted_coverage', 0) * 100:.0f}%")
    
    # Display feedback summary
    st.subheader("Feedback Summary")
    st.markdown(result["feedback"]["summary"])
    
    # Display student review and expert review side by side
    st.subheader("Review Comparison")
    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.markdown("#### Your Review")
        st.markdown(st.session_state.student_review)
    
    with comp_col2:
        st.markdown("#### Expert Review")
        st.markdown(st.session_state.current_exercise["expert_review"])
    
    # Display strengths and improvements
    strength_col, improve_col = st.columns(2)
    with strength_col:
        st.subheader("Strengths")
        for strength in result["comparison"]["strengths"]:
            st.markdown(f"✅ {strength}")
    
    with improve_col:
        st.subheader("Areas for Improvement")
        for improvement in result["comparison"]["improvements"]:
            st.markdown(f"📈 {improvement}")
    
    # Display missed issues with feedback
    if result["feedback"]["missed_issues_feedback"]:
        st.subheader("Issues You Missed")
        for feedback in result["feedback"]["missed_issues_feedback"]:
            st.markdown(f"- {feedback}")
    
    # Display learning tips
    if result["feedback"]["learning_tips"]:
        st.subheader("Learning Tips")
        for tip in result["feedback"]["learning_tips"]:
            st.markdown(f"💡 {tip}")
    
    # Display next exercise recommendation
    st.subheader("Next Steps")
    agent = st.session_state.agent
    recommendation = agent.recommend_next_exercise()
    
    st.markdown(f"**Recommendation:** {recommendation['rationale']}")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Try Another Exercise"):
            # Reset exercise and review data
            st.session_state.current_exercise = None
            st.session_state.review_result = None
            st.session_state.student_review = ""
            # Generate a new exercise with recommended parameters
            generate_new_exercise(
                recommendation["difficulty"],
                recommendation["focus_areas"]
            )
    
    with col2:
        if st.button("View Learning Progress"):
            st.session_state.step = 4
    
    with col3:
        if st.button("Return to Start"):
            st.session_state.step = 1

# Render learning progress step
def render_learning_progress(section_num: int):
    """
    Render the learning progress and statistics step.
    
    Args:
        section_num (int): Section number for header
    """
    st.header(f"{section_num}. Your Learning Progress")
    
    agent = st.session_state.agent
    learning_summary = agent.generate_learning_summary()
    
    # Display overview stats
    st.subheader("Learning Overview")
    st.markdown(learning_summary["status"])
    
    # Create a progress metrics dashboard
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Level", learning_summary["level"].capitalize())
    with metric_col2:
        st.metric("Exercises Completed", learning_summary["completed_exercises"])
    with metric_col3:
        st.metric("Average Score", f"{learning_summary.get('average_score', 0)}/100")
    with metric_col4:
        st.metric("Issue Detection Rate", f"{learning_summary.get('catch_rate', 0)}%")
    
    # Display strengths and weaknesses
    strength_col, weakness_col = st.columns(2)
    with strength_col:
        st.subheader("Your Strengths")
        strengths = learning_summary.get("strengths", [])
        if strengths:
            for strength in strengths:
                st.markdown(f"✅ {strength}")
        else:
            st.info("Complete more exercises to identify your strengths.")
    
    with weakness_col:
        st.subheader("Areas for Improvement")
        weaknesses = learning_summary.get("weaknesses", [])
        if weaknesses:
            for weakness in weaknesses:
                st.markdown(f"🔍 {weakness}")
        else:
            st.info("Complete more exercises to identify areas for improvement.")
    
    # Display recommendations
    st.subheader("Recommendations")
    st.markdown(learning_summary.get("recommendations", ""))
    st.markdown(learning_summary.get("next_steps", ""))
    
    # If we have enough data, show a progress chart
    scores_over_time = learning_summary.get("scores_over_time", [])
    if len(scores_over_time) >= 2:
        st.subheader("Progress Over Time")
        
        # Create dataframe and plot
        progress_df = pd.DataFrame(scores_over_time)
        progress_df["date"] = progress_df["timestamp"].apply(
            lambda x: time.strftime('%Y-%m-%d %H:%M', time.localtime(x))
        )
        
        st.line_chart(progress_df, x="date", y="score")
    
    # Navigation buttons
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Back to Results"):
            st.session_state.step = 3
    with col2:
        if st.button("Start New Exercise"):
            # Reset exercise and review data
            st.session_state.current_exercise = None
            st.session_state.review_result = None
            st.session_state.student_review = ""
            # Go to step 1
            st.session_state.step = 1

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
    st.title("🧠 AI-Enhanced Peer Code Review Tutorial System")
    
    # Check and display system status
    system_ready = display_system_status()
    
    # If system is not ready, don't proceed further
    if not system_ready and not st.session_state.system_ready:
        st.info("Please address the system requirements above to continue.")
        return
    
    # Organize sections with proper numbering for main content
    section_num = 1
    
    # LLM Model Setup Section
    render_llm_setup_section(section_num)
    section_num += 1
    
    # Handle different steps of the tutorial
    if not st.session_state.model_initialized:
        st.warning("Please initialize an Ollama model before continuing.")
    elif st.session_state.step == 1:
        # Step 1: Introduction and exercise selection
        render_introduction_step(section_num)
    elif st.session_state.step == 2:
        # Step 2: Code review exercise
        render_code_review_step(section_num)
    elif st.session_state.step == 3:
        # Step 3: Feedback and results
        render_feedback_step(section_num)
    elif st.session_state.step == 4:
        # Step 4: Learning progress
        render_learning_progress(section_num)

# Run the application
if __name__ == "__main__":
    main()