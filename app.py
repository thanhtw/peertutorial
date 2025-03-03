import streamlit as st
import subprocess
import os
import tempfile
import json
from modules.ai_agent import (
    get_code_review_knowledge,
    generate_code_snippet,
    get_ai_review,
    compare_reviews
)
from modules.compiler import compile_java
from modules.checkstyle import run_checkstyle

# Set page configuration
st.set_page_config(
    page_title="Peer Code Review Tutorial",
    page_icon="📝",
    layout="wide"
)

# Initialize session state
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

# Page header
st.title("🧠 Peer Code Review Tutorial System")

# Function to move to the next step
def next_step():
    st.session_state.step += 1

# Function to restart the tutorial
def restart_tutorial():
    st.session_state.step = 1
    st.session_state.code_snippet = ""
    st.session_state.compile_results = None
    st.session_state.checkstyle_results = None
    st.session_state.ai_review = ""
    st.session_state.student_review = ""
    st.session_state.comparison = None

# Function to generate new code snippet
def generate_new_snippet():
    with st.spinner("Generating code snippet and analyzing..."):
        # Generate code snippet using AI
        code_snippet = generate_code_snippet()
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
        st.session_state.ai_review = ai_review
        
        # Clean up temporary file
        os.unlink(tmp_path)
    
    next_step()

# Function to submit student review
def submit_review():
    with st.spinner("Analyzing your review..."):
        # Get comparison between AI and student reviews
        comparison = compare_reviews(
            st.session_state.ai_review,
            st.session_state.student_review
        )
        st.session_state.comparison = comparison
    
    next_step()

# Step 1: Introduction and knowledge about peer code review
if st.session_state.step == 1:
    st.header("Welcome to the Peer Code Review Tutorial")
    
    # Load knowledge about peer code review if not already loaded
    if not st.session_state.knowledge:
        with st.spinner("Loading code review knowledge..."):
            knowledge = get_code_review_knowledge()
            st.session_state.knowledge = knowledge
    
    st.markdown(st.session_state.knowledge)
    
    st.subheader("How this tutorial works:")
    st.write("""
    1. You'll be shown a Java code snippet with potential issues
    2. The system will compile the code and run style checks
    3. Review the code and write your comments
    4. Compare your review with an AI-generated review
    5. Receive feedback on your reviewing skills
    """)
    
    st.button("Start Practice", on_click=generate_new_snippet)

# Step 2: Code snippet review
elif st.session_state.step == 2:
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

# Step 3: Comparison and feedback
elif st.session_state.step == 3:
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