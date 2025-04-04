import streamlit as st
import openai
import time
import tempfile

st.write("OpenAI package version:", openai.__version__)

# Set your API key from secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("Engineering PDF Assistant")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
user_question = st.text_input("Ask a question about the PDF")

if uploaded_file and user_question:
    with st.spinner("Uploading file..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        with open(tmp_file_path, "rb") as f:
            file_response = openai.files.create(file=f, purpose='assistants')
        file_id = file_response.id

    with st.spinner("Creating assistant..."):
        
    from openai.types.beta.assistant_create_params import Tool
    
    assistant = openai.beta.assistants.create(
        name="PDF Assistant",
        instructions="You are an engineering assistant. Use the uploaded document to answer questions and cite specific sections where appropriate.",
        model="gpt-4-turbo",
        tools=[Tool(type="retrieval")],
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

    with st.spinner("Waiting for response..."):
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
                break
            time.sleep(1)

    messages = openai.beta.threads.messages.list(thread_id=thread.id)
    for msg in messages.data:
        st.markdown(msg.content[0].text.value)

