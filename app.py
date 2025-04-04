import streamlit as st
import openai
import time
import tempfile

# 👇 Use correct import for 'tools' with SDK v1.70.0
from openai.types.beta.assistant_create_params import ToolAssistantTools

# Load OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("Engineering PDF Assistant")
st.caption("Ask a question about your uploaded engineering PDF document.")

# Debug: show SDK version
st.write("OpenAI SDK version:", openai.__version__)

# File uploader and question input
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
user_question = st.text_input("Ask a question about the PDF")

# Only run once both file and question are provided
if uploaded_file and user_question:
    with st.spinner("Uploading file to OpenAI..."):
        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Upload file to OpenAI
        with open(tmp_path, "rb") as f:
            file_response = openai.files.create(file=f, purpose="assistants")
        file_id = file_response.id
        st.write("File uploaded successfully. File ID:", file_id)

    with st.spinner("Creating assistant..."):
        try:
            assistant = openai.beta.assistants.create(
                name="PDF Assistant",
                instructions="You are an engineering assistant. Use the uploaded document to answer questions and cite specific sections where appropriate.",
                model="gpt-4-turbo",
                tools=[ToolAssistantTools(type="retrieval")],  # ✅ Correct format for SDK v1.70.0
                file_ids=[file_id]
            )
        except Exception as e:
            st.error(f"Assistant creation failed: {e}")
            st.stop()

    with st.spinner("Creating conversation thread..."):
        thread = openai.beta.threads.create()

    with st.spinner("Sending your question to the assistant..."):
        openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_question
        )

    with st.spinner("Waiting for assistant's response..."):
        run = openai.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )

        while True:
            run_status = openai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run_status.status == "completed":
                break
            elif run_status.status == "failed":
                st.error("The assistant failed to process your request.")
                st.stop()
            time.sleep(1)

    # Display the assistant's response
    messages = openai.beta.threads.messages.list(thread_id=thread.id)
    for msg in reversed(messages.data):  # Show assistant's message last
        if msg.role == "assistant":
            st.markdown("### Assistant Response")
            st.markdown(msg.content[0].text.value)



