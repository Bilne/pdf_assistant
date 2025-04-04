import streamlit as st
from openai import OpenAI

# Set your OpenAI API key (or use environment variable)
OPENAI_API_KEY = "your-api-key-here"  # Replace with your key or use os.environ

client = OpenAI(api_key=OPENAI_API_KEY)

st.title("💬 Chat with GPT-4 (Streaming)")
st.subheader("OpenAI API via Streamlit")

user_input = st.text_input("You:", placeholder="Ask me anything...")

if user_input:
    # Display a placeholder for streaming response
    response_placeholder = st.empty()
    full_response = ""

    with client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": user_input}],
        stream=True
    ) as stream:
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                response_placeholder.markdown(full_response)


