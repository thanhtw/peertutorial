"""
Function to render the Ollama model setup section.
This is based on langchain's _render_llm_setup_section but simplified to only use Ollama.
"""

import streamlit as st
import requests
import time
import os
from typing import Dict, List, Any, Optional
from modules.ai_agent import is_ollama_running, pull_model, set_ollama_model, OLLAMA_URL

def _render_llm_setup_section(section_num: int):
    """
    Render LLM setup section focused on Ollama models.
    
    Args:
        section_num (int): Section number for header
    """
    st.header(f"{section_num}. Setup Ollama LLM Model")
    
    if not is_ollama_running():
        st.error("Ollama server is not running. Please start it from the sidebar.")
        return
    
    # Show available models
    st.write("### Available Ollama Models")
    st.write("Select a model from the dropdown. Models with ✅ are already pulled and ready to use.")

    # Fetch available models
    available_models = _get_ollama_models()
    
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
                            if pull_model(model_id):
                                st.success(f"Successfully pulled {selected_model['name']}")
                                # Force refresh of models
                                st.session_state.available_models = _get_ollama_models()
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
                            # Set the model in the ai_agent module
                            if set_ollama_model(model_id):
                                st.session_state.model_initialized = True
                                st.session_state.selected_model = model_id
                                st.session_state.initialization_message = f"Using {selected_model['name']}"
                                st.success(f"Successfully initialized {selected_model['name']}!")
                            else:
                                st.error(f"Failed to initialize {selected_model['name']}")
            
            # Show model status if initialized
            if st.session_state.get("model_initialized", False) and st.session_state.get("selected_model") == model_id:
                st.success(f"✅ {selected_model['name']} is currently initialized and ready to use")
                
                # Show current parameters
                params = st.session_state.get("model_params", {})
                if params:
                    st.write("Current parameters:")
                    params_str = ", ".join([f"{k}: {v}" for k, v in params.items()])
                    st.code(params_str)

def _get_ollama_models() -> List[Dict[str, Any]]:
    """
    Get a list of available Ollama models.
    
    Returns:
        List of model information dictionaries
    """
    # Default models that can be pulled
    default_models = [
        {"id": "llama3:1b", "name": "Llama 3 (1B)", "description": "Fast, lightweight Llama 3 model", "pulled": False},
        {"id": "llama3:8b", "name": "Llama 3 (8B)", "description": "Good balance of quality and speed", "pulled": False},
        {"id": "deepseek-coder:6.7b", "name": "DeepSeek Coder (6.7B)", "description": "Specialized for code review", "pulled": False},
        {"id": "phi3:mini", "name": "Phi-3 Mini", "description": "Compact model with good reasoning", "pulled": False},
        {"id": "gemma:2b", "name": "Gemma 2B", "description": "Google's efficient small model", "pulled": False}
    ]
    
    try:
        # Try to get list of already pulled models from Ollama API
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        
        if response.status_code == 200:
            pulled_models = response.json().get("models", [])
            pulled_ids = [model["name"] for model in pulled_models]
            
            # Mark models as pulled if they exist locally
            for model in default_models:
                if model["id"] in pulled_ids:
                    model["pulled"] = True
            
            # Add any pulled models that aren't in our standard list
            for pulled_model in pulled_models:
                model_id = pulled_model["name"]
                if not any(model["id"] == model_id for model in default_models):
                    # Extract size information
                    size_str = pulled_model.get("size", "Unknown")
                    # Add to the list
                    default_models.append({
                        "id": model_id,
                        "name": model_id,
                        "description": f"Size: {size_str}",
                        "pulled": True
                    })
        
        # Store in session state for reuse
        st.session_state.available_models = default_models
        return default_models
            
    except Exception as e:
        print(f"Error connecting to Ollama: {str(e)}")
        # Return list with local models marked as pulled
        return default_models