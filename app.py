import streamlit as st
from openai import OpenAI

# Check for API key in secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.error("🚨 API key not found in Streamlit secrets!")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# App layout
st.title("💬 Chat with GPT (Streaming)")
st.subheader("Ask a question and watch the response stream in real-time")

# Model selector
model = st.selectbox("Choose a model:", ["gpt-4", "gpt-3.5-turbo"])

# User input
user_input = st.text_input("You:", placeholder="Ask me anything...")

# Streamed response
if user_input:
    response_placeholder = st.empty()
    full_response = ""

    with client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_input}],
        stream=True
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                full_response += delta.content
                response_placeholder.markdown(full_response)