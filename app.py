
import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Check secrets
if "openai_api_key" not in st.secrets:
    st.error("❌ Add your OpenAI API key to .streamlit/secrets.toml")
    st.stop()

client = OpenAI(api_key=st.secrets["openai_api_key"])

st.title("📄 Multi-PDF Chat with RAG + GPT-4")
st.caption("Upload PDFs, ask questions, and get GPT-4 answers using document retrieval.")

# Upload multiple PDFs
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

# Extract and process text
file_texts = []

if uploaded_files:
    for file in uploaded_files:
        text = ""
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        file_texts.append(f"--- {file.name} ---\n{text}")

    combined_text = "\n\n".join(file_texts)
    st.success(f"✅ {len(uploaded_files)} file(s) processed.")
    
    with st.expander("📄 Preview extracted text"):
        st.write(combined_text[:2000] + "..." if len(combined_text) > 2000 else combined_text)

    # --- RAG Setup ---
    st.info("🔍 Creating embeddings and vector index...")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(combined_text)]

    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_api_key"])
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    st.success("✅ Vector index created. Ask your question below!")

    # User question input
    user_input = st.text_input("💬 Ask a question about the documents:")

    if user_input:
        relevant_docs = retriever.get_relevant_documents(user_input)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        response_placeholder = st.empty()
        full_response = ""

        with client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer using only the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_input}"},
            ],
            stream=True
        ) as stream:
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(full_response)

else:
    st.info("📥 Upload PDF files to get started.")
