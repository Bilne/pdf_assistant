import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF

# Load API key from secrets
if "openai_api_key" not in st.secrets:
    st.error("🚨 API key not found in Streamlit secrets!")
    st.stop()

client = OpenAI(api_key=st.secrets["openai_api_key"])

st.title("📄 Ask Questions About a File")
st.subheader("Upload a PDF or TXT file and chat with GPT-4")

# File upload
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

# Read file content
file_text = ""

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                file_text += page.get_text()
    elif uploaded_file.type == "text/plain":
        file_text = uploaded_file.read().decode("utf-8")

    st.success("✅ File uploaded and text extracted")

    # Show preview of file text
    with st.expander("📄 Preview Extracted Text"):
        st.write(file_text[:1000] + "...")  # Limit preview to 1000 characters

    # User question
    user_input = st.text_input("Ask a question about the file:")

    if user_input:
        response_placeholder = st.empty()
        full_response = ""

        with client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You're a helpful assistant answering questions about the uploaded document."},
                {"role": "user", "content": f"The document says:\n\n{file_text[:4000]}"},
                {"role": "user", "content": user_input},
            ],
            stream=True
        ) as stream:
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(full_response)

else:
    st.info("👆 Please upload a PDF or TXT file to begin.")
