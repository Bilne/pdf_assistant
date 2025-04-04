import streamlit as st
import openai
import time
import tempfile
import mimetypes

# Load OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("Engineering PDF Assistant")
st.caption("Ask a question about your uploaded engineering PDF document.")
st.write("OpenAI SDK version:", openai.__version__)

# File uploader and user question
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
user_question = st.text_input("Ask a question about the PDF")

if uploaded_file and user_question:
    with st.spinner("Saving and uploading file to OpenAI..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            # Detect content type for upload
            content_type = mimetypes.guess_type(uploaded_file.name)[0] or "application/pdf"
            with open(tmp_path, "rb") as f:
                file_response = openai.files.create(
                    file=(uploaded_file.name, f, content_type),
                    purpose="assistants"
                )
            file_id = file_response.id
            st.success(f"File uploaded successfully. File ID: {file_id}")
        except Exception as e:
            st.error(f"File upload failed: {e}")
            st.stop()

    with st.spinner("Creating assistant..."):
        try:
            assistant = openai.beta.assistants.create(
                name="PDF Assistant",
                instructions=(
                    "You are an engineering assistant. Use the uploaded PDF to answer questions. "
                    "Provide clear and helpful responses. If applicable, cite relevant sections."
                ),
                model="gpt-4-turbo",
                tools=[{"type": "file_search"}]  # ✅ Updated tool type
            )
        except Exception as e:
            st.error(f"Assistant creation failed: {e}")
            st.stop()

    with st.spinner("Creating conversation thread..."):
        try:
            thread = openai.beta.threads.create()
        except Exception as e:
            st.error(f"Thread creation failed: {e}")
            st.stop()

    with st.spinner("Sending your question..."):
        try:
            openai.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_question
            )
        except Exception as e:
            st.error(f"Message send failed: {e}")
            st.stop()

    with st.spinner("Waiting for assistant's response..."):
        try:
            # ✅ Attach the file here in the run
            run = openai.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id,
                file_ids=[file_id]
            )

            while True:
                run_status = openai.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                if run_status.status == "completed":
                    break
                elif run_status.status == "failed":
                    st.error("The assistant failed to complete your request.")
                    st.stop()
                time.sleep(1)
        except Exception as e:
            st.error(f"Run failed: {e}")
            st.stop()

    with st.spinner("Fetching assistant response..."):
        try:
            messages = openai.beta.threads.messages.list(thread_id=thread.id)
            for msg in reversed(messages.data):
                if msg.role == "assistant":
                    st.markdown("### Assistant Response")
                    st.markdown(msg.content[0].text.value)
        except Exception as e:
            st.error(f"Failed to retrieve messages: {e}")


