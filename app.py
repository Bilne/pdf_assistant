import streamlit as st
import openai
import time
import tempfile

# Set your OpenAI API key securely
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("Engineering PDF Assistant")
st.caption("Ask a question about your uploaded engineering PDF document.")

# Show OpenAI SDK version (for debug)
st.write("OpenAI SDK version:", openai.__version__)

# Upload the PDF and ask a question
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
user_question = st.text_input("Ask a question about the PDF")

# Process only if both file and question are present
if uploaded_file and user_question:
    with st.spinner("Uploading file..."):
        # Save the uploaded file to a temp location so it can be passed to OpenAI
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Upload to OpenAI for use with assistant
        with open(tmp_path, "rb") as f:
            file_response = openai.files.create(file=f, purpose="assistants")
        file_id = file_response.id

    st.write("File ID:", file_id, type(file_id))

    with st.spinner("Creating assistant..."):
        assistant = openai.beta.assistants.create(
            name="PDF Assistant",
            instructions="You are an engineering assistant. Use the uploaded document to answer questions and cite specific sections where appropriate.",
            model="gpt-4-turbo",
            tools=[{"type": "retrieval"}],  # ✅ Valid for SDK v1.70.0
            file_ids=[file_id]
        )

    with st.spinner("Creating thread..."):
        thread = openai.beta.threads.create()

    with st.spinner("Sending question..."):
        openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_question
        )

    with st.spinner("Waiting for assistant response..."):
        run = openai.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )

        # Poll for completion
        while True:
            run_status = openai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run_status.status == "completed":
                break
            elif run_status.status == "failed":
                st.error("The assistant failed to process your request.")
                st.stop()
            time.sleep(1)

    # Display assistant response
    messages = openai.beta.threads.messages.list(thread_id=thread.id)
    for msg in reversed(messages.data):  # Newest last
        if msg.role == "assistant":
            st.markdown("### Assistant Response")
            st.markdown(msg.content[0].text.value)


