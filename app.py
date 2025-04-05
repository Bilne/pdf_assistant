import streamlit as st
import openai

# Load OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai"]["api_key"]

st.title("Chat with OpenAI")

# User input
user_input = st.text_input("Ask a question:", "")

# Optional controls
temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.7)
model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])

if user_input:
    with st.spinner("Thinking..."):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input}
                ],
                temperature=temperature
            )
            st.markdown("**Response:**")
            st.write(response["choices"][0]["message"]["content"])
        except Exception as e:
            st.error(f"API Error: {e}")

