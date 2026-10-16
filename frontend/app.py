import sys
import os

# Must be before any backend imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st

try:
    from backend.pipeline import rag_pipeline
except ModuleNotFoundError as e:
    st.error("❌ Could not import backend.pipeline.")
    raise e
st.set_page_config(page_title="Medical Research Q&A App", layout="wide")
st.title("📚 Medical Research Q&A")
st.write("Ask questions about Medical research papers and get natural language answers.")

query = st.text_input("Enter your question:")
image_file = st.file_uploader("Upload a medical image (optional)", type=["jpg", "jpeg", "png"])

if query:
    with st.spinner("Retrieving and generating answer..."):
        try:
            image_filename = image_file.name if image_file else None
            answer = rag_pipeline(query, image_file=image_file, image_filename=image_filename)

            st.subheader("Answer")
            st.markdown(answer)

        except Exception as e:
            st.error(f"❌ Error running pipeline: {e}")