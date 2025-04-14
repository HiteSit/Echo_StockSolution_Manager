import streamlit as st
import requests

st.title("CSV Upload App")

backend_url = "http://localhost:8000/upload_csv"

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    st.write(f"Selected file: {uploaded_file.name}")
    if st.button("Upload CSV"):
        with st.spinner("Uploading..."):
            files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
            try:
                response = requests.post(backend_url, files=files)
                if response.status_code == 200:
                    st.success("Upload and concatenation successful!")
                else:
                    st.error(f"Upload failed: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"Error: {e}")
