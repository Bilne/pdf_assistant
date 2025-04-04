import streamlit as st
from openai import OpenAI

# Get API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("💬 Chat with GPT-4 (Streaming)")
st.subheader("OpenAI API via Streamlit")

user_input = st.text_input("You:", placeholder="Ask me anything...")

if user_input:
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
