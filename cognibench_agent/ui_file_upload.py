import logging

import streamlit as st


def render_file_uploader():
    """Renders the file uploader and saves uploaded files to temp dir with validation and sanitization."""
    logger = logging.getLogger("frontend")
    st.header("Upload Raw RLHF JSON Data file(s)")
    uploaded_files = st.file_uploader(
        "Select CogniBench JSON batch file(s)",
        type=["json"],
        accept_multiple_files=True,
        help="Upload one or more JSON files containing tasks for evaluation.",
        key="workflow_file_uploader",
    )

    if uploaded_files:
        uploaded_file_names = [f.name for f in uploaded_files]
        current_upload_key = tuple(sorted(uploaded_file_names))

        if (
            st.session_state.last_uploaded_files_key != current_upload_key
            or not st.session_state.uploaded_files_info
        ):
            with st.spinner("Processing uploaded files..."):
                logger.info(f"Processing {len(uploaded_files)} uploaded files...")
                st.session_state.uploaded_files_info = []
                temp_dir = st.session_state.temp_dir_path
                for uploaded_file in uploaded_files:
                    try:
                        # Validate JSON structure before saving
                        import json

                        file_bytes = uploaded_file.getvalue()
                        try:
                            data = json.loads(file_bytes.decode("utf-8"))
                            # Accept either a list of objects or a single object
                            if isinstance(data, dict):
                                data = [data]
                            if not isinstance(data, list):
                                st.warning(
                                    f"File {uploaded_file.name} is not a list or object. It may not be a valid evaluation file."
                                )
                                logger.warning(
                                    f"File {uploaded_file.name} is not a list or object."
                                )
                            # Check for required fields in the first object, but do not block upload
                            required_fields = [
                                "prompt_id",
                                "prompt_content",
                                "model_response_id",
                                "model_name",
                                "model_response_text",
                                "ideal_response_id",
                                "ideal_response_text",
                                "correct_answer",
                            ]
                            present_keys = list(data[0].keys()) if data and isinstance(data[0], dict) else []
                            logger.info(f"File {uploaded_file.name} first object keys: {present_keys}")
                            missing_fields = [field for field in required_fields if field not in present_keys]
                            if len(missing_fields) == len(required_fields):
                                st.warning(
                                    f"File {uploaded_file.name} does not match the expected evaluation schema. Present keys: {present_keys}"
                                )
                                logger.warning(
                                    f"File {uploaded_file.name} does not match the expected evaluation schema. Present keys: {present_keys}"
                                )
                        except Exception as ve:
                            st.error(
                                f"File {uploaded_file.name} is not a valid JSON: {ve}"
                            )
                            logger.error(
                                f"File {uploaded_file.name} failed JSON parsing: {ve}"
                            )
                            continue

                        dest_path = temp_dir / uploaded_file.name
                        with open(dest_path, "wb") as f:
                            f.write(file_bytes)
                        # Sanitize file name
                        safe_name = "".join(
                            c
                            for c in uploaded_file.name
                            if c.isalnum() or c in ("-", "_", ".")
                        )
                        st.session_state.uploaded_files_info.append(
                            {"name": safe_name, "path": str(dest_path)}
                        )
                        logger.info(
                            f"Saved uploaded file to temporary path: {dest_path}"
                        )
                    except Exception as e:
                        st.error(f"Error saving file {uploaded_file.name}: {e}")
                        logger.error(f"Error saving file {uploaded_file.name}: {e}")

                st.session_state.last_uploaded_files_key = current_upload_key
                logger.info(
                    f"Finished processing uploads. {len(st.session_state.uploaded_files_info)} files ready."
                )
                st.rerun()

        st.write(f"Using {len(st.session_state.uploaded_files_info)} uploaded file(s):")
        for file_info in st.session_state.uploaded_files_info:
            st.write(f"- {file_info['name']}")
    else:
        st.info("Please upload at least one batch file.")
        if st.session_state.uploaded_files_info:
            st.session_state.uploaded_files_info = []
            st.session_state.last_uploaded_files_key = None
            st.session_state.uploaded_files_info = []
            st.session_state.last_uploaded_files_key = None
