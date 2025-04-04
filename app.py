
import streamlit as st
import fitz  # PyMuPDF
import tempfile
from openai import OpenAI

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# API Key setup
if "openai_api_key" not in st.secrets:
    st.error("❌ OpenAI API key missing from secrets.toml.")
    st.stop()

client = OpenAI(api_key=st.secrets["openai_api_key"])

st.title("📄 Multi-PDF Chat with RAG + GPT-4")
st.caption("Upload PDFs, embed their content, and ask context-aware questions.")

uploaded_files = st.file_uploader("📥 Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

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

    # RAG setup
    st.info("🔍 Creating vector store with Chroma...")

    # 1. Chunk text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = [Document(page_content=chunk) for chunk in splitter.split_text(combined_text)]

    # 2. Embed and index
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_api_key"])
    persist_dir = tempfile.mkdtemp()
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    st.success("✅ Chroma index created. Ask a question!")

    user_input = st.text_input("💬 Ask a question about the documents:")

    if user_input:
        docs = retriever.get_relevant_documents(user_input)
        context = "\n\n".join([doc.page_content for doc in docs])

        response_placeholder = st.empty()
        full_response = ""

        with client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You're a helpful assistant. Use the following context to answer the user's question."},
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

