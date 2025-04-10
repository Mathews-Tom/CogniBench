import streamlit as st
import os

st.set_page_config(layout="wide")

st.title("CogniBench Evaluation Runner")

# --- Phase 1: Input Selection ---
st.header("1. Upload Batch File(s)")

uploaded_files = st.file_uploader(
    "Select CogniBench JSON batch file(s)",
    type=['json'],
    accept_multiple_files=True,
    help="Upload one or more JSON files containing tasks for evaluation."
)

if uploaded_files:
    st.write(f"Uploaded {len(uploaded_files)} file(s):")
    for uploaded_file in uploaded_files:
        st.write(f"- {uploaded_file.name}")
else:
    st.info("Please upload at least one batch file.")

# Placeholder for Folder Picker (if implemented later)
# st.write("*(Folder selection coming soon)*")

# --- Phase 1.5: Configuration Options ---
st.header("2. Configure Evaluation")

# Define available models based on the plan
AVAILABLE_MODELS = {
    "openai": {
        "GPT-4O": "gpt-4o",
        "GPT-4 Turbo": "gpt-4-turbo",
        "GPT-4": "gpt-4",
        "GPT-3.5 Turbo": "gpt-3.5-turbo",
    },
    "anthropic": {
        "Claude 3.5 Haiku": "claude-3-5-haiku-latest", # Placeholder ID
        "Claude 3.5 Sonnet": "claude-3-5-sonnet-20240620",
        "Claude 3.7 Sonnet": "claude-3-7-sonnet-latest", # Placeholder ID
        "Claude 3 Opus": "claude-3-opus-20240229",
    },
    "google": {
        "Gemini 1.5 Flash": "gemini-1.5-flash-latest",
        "Gemini 1.5 Pro": "gemini-1.5-pro-latest",
        "Gemini 2.0 Flash": "gemini-2.0-flash", # Placeholder ID
        "Gemini 2.5 Pro Preview": "gemini-2.5-pro-preview-03-25", # Placeholder ID
    },
    "local_llm": {
        # Add local models here if needed
    }
}

# Define available prompt templates
PROMPT_TEMPLATES_DIR = "../prompts" # Relative to app.py location
try:
    # List files relative to the CogniBench root, then construct relative path for display
    prompt_files = [f for f in os.listdir(os.path.join(os.path.dirname(__file__), PROMPT_TEMPLATES_DIR)) if f.endswith('.txt')]
    AVAILABLE_TEMPLATES = {os.path.basename(f): os.path.join(PROMPT_TEMPLATES_DIR, f).replace("../", "") for f in prompt_files}
except FileNotFoundError:
    st.error(f"Prompt templates directory not found at expected location: {os.path.abspath(os.path.join(os.path.dirname(__file__), PROMPT_TEMPLATES_DIR))}")
    AVAILABLE_TEMPLATES = {}


# Initialize session state
if 'selected_provider' not in st.session_state:
    st.session_state.selected_provider = list(AVAILABLE_MODELS.keys())[0] # Default to openai
if 'selected_model_name' not in st.session_state:
    st.session_state.selected_model_name = list(AVAILABLE_MODELS[st.session_state.selected_provider].keys())[0]
if 'selected_template_name' not in st.session_state:
    st.session_state.selected_template_name = list(AVAILABLE_TEMPLATES.keys())[0] if AVAILABLE_TEMPLATES else None
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

# --- Configuration Widgets ---
col_config1, col_config2 = st.columns(2)

with col_config1:
    # Provider Selection
    st.session_state.selected_provider = st.selectbox(
        "Select LLM Provider",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_provider),
        key="provider_select" # Use key to avoid issues with re-rendering
    )

    # Model Selection (dynamic based on provider)
    available_model_names = list(AVAILABLE_MODELS[st.session_state.selected_provider].keys())
    # Ensure the previously selected model is still valid for the new provider, else default
    current_model_index = 0
    if st.session_state.selected_model_name in available_model_names:
        current_model_index = available_model_names.index(st.session_state.selected_model_name)
    else:
        st.session_state.selected_model_name = available_model_names[0] # Default to first model of new provider

    st.session_state.selected_model_name = st.selectbox(
        "Select Judge Model",
        options=available_model_names,
        index=current_model_index,
        key="model_select"
    )

with col_config2:
    # API Key Input
    st.session_state.api_key = st.text_input(
        "API Key (Optional)",
        type="password",
        placeholder="Leave blank to use environment variable",
        value=st.session_state.api_key,
        key="api_key_input"
    )

    # Prompt Template Selection
    if AVAILABLE_TEMPLATES:
        st.session_state.selected_template_name = st.selectbox(
            "Select Evaluation Prompt Template",
            options=list(AVAILABLE_TEMPLATES.keys()),
            index=list(AVAILABLE_TEMPLATES.keys()).index(st.session_state.selected_template_name) if st.session_state.selected_template_name in AVAILABLE_TEMPLATES else 0,
            key="template_select"
        )
    else:
        st.warning("No prompt templates found. Please add templates to the 'prompts/' directory.")
        st.session_state.selected_template_name = None


# --- Display Current Configuration ---
st.subheader("Current Configuration:")
if st.session_state.selected_template_name:
    selected_model_api_id = AVAILABLE_MODELS[st.session_state.selected_provider][st.session_state.selected_model_name]
    selected_template_path = AVAILABLE_TEMPLATES[st.session_state.selected_template_name]
    api_key_status = "Provided" if st.session_state.api_key else "Using Environment Variable"

    st.json({
        "Provider": st.session_state.selected_provider,
        "Model Name": st.session_state.selected_model_name,
        "Model API ID": selected_model_api_id,
        "API Key Status": api_key_status,
        "Prompt Template": selected_template_path
    })
else:
    st.warning("Configuration incomplete due to missing prompt templates.")

# --- Phase 2: Run Evaluation ---
st.header("3. Run Evaluation")

col1, col2 = st.columns(2)

with col1:
    run_button = st.button("🚀 Run Evaluation", type="primary", disabled=not uploaded_files)

with col2:
    clear_cache_button = st.button("🧹 Clear LLM Cache")

if run_button:
    st.info("Evaluation running... (Implementation pending in Phase 2)")
    # TODO: Implement subprocess call and progress monitoring

if clear_cache_button:
    st.info("Clearing cache... (Implementation pending in Utility Features)")
    # TODO: Implement cache clearing logic

# --- Phase 3: Visualize Results (Placeholder) ---
st.header("4. Results")
st.info("Evaluation results will be displayed here after running (Phase 3).")
# TODO: Add results visualization