import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.pdfextractor import text_extractor

from pypdf import PdfReader
import streamlit as st

# Load Environment variables
from dotenv import load_dotenv
load_dotenv()


st.title(":green[RAG Chatbot]")
st.subheader("Upload the PDF in the sidebar")

st.sidebar.subheader("Upload the PDF")
file_upload = st.sidebar.file_uploader("Uplaod the PDF file you want to know details about", type="pdf")


if file_upload:
    file_text = text_extractor(file_upload)


    # Configure the Gemini key
    key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=key)
    llm_model = genai.GenerativeModel("gemini-3-flash-preview")

    # Configure Embedding
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Split the text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(file_text)

    # Create the FAISS vector
    vector_store = FAISS.from_texts(chunks, embedding_model)

    # Configure the retriever
    retriever = vector_store.as_retriever(search_kwargs={"k":3})


    def generate_response(query:str)->str:
        retrival_docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in retrival_docs])
        prompt = f"""
                You are helpful assistant for answering questions based on teh following context, ising RAG:
                {context}
                user query: {query}
                """

        content = llm_model.generate_content(prompt)
        return content.text if hasattr(content, "text") else content.candidates[0].content.parts[0].text


    # Initialize chat if there is no history
    if 'history' not in st.session_state:
        st.session_state.history = []
    # Display the History
    for msg in st.session_state.history:
        if msg['role'] == 'user':
            st.write(f':green[User:] :blue[{msg["text"]}]')
        else:
            st.write(f':orange[Chatbot:] {msg["text"]}')
    # Input from the user (Using Streamlit Form)
    with st.form('Chat Form', clear_on_submit=True):
        user_input = st.text_input('Enter Your Text Here:')
        send = st.form_submit_button('Send')
    # Start the conversation and append the output and query in history
    if user_input and send:
        st.session_state.history.append({"role": 'user', "text": user_input})
        model_output = generate_response(user_input)
        st.session_state.history.append({'role': 'chatbot', 'text': model_output})
        st.rerun()

