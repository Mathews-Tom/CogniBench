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

# --- Phase 1.5: Configuration Options (Placeholder) ---
st.header("2. Configure Evaluation")
st.info("Configuration options will appear here (Phase 1.5).")
# TODO: Add configuration widgets (LLM provider, model, API key, template)

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