import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import wikipediaapi
import PyPDF2 as pdf
from langchain_core.tools import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, GoogleSearchAPIWrapper
from dotenv import load_dotenv
load_dotenv()

# Importing API keys from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.environ.get("GOOGLE_CSE_ID")
# Streamlit
st.title("RAG application")

# Sidebar for user to upload PDF files
uploaded_files = st.sidebar.file_uploader("Upload your PDF files", accept_multiple_files=True, type=["pdf"])

# Function to save uploaded files
def save_uploaded_files(files):
    if not os.path.exists("source_documents"):
        os.makedirs("source_documents")
    for file in files:
        with open(os.path.join("source_documents", file.name), "wb") as f:
            f.write(file.getbuffer())

# Function to create vector embeddings from saved PDF files
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        documents = []
        source_dir = "source_documents"
        for filename in os.listdir(source_dir):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(source_dir, filename))
                documents.extend(loader.load())
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=220)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(documents)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

def google_output(user_input):
    search = GoogleSearchAPIWrapper()
    tool = Tool(
        name="google_search",
        description="Search Google for recent results.",
        func=search.run,
    )
    google_result = tool.run(user_input)
    return google_result

def fetch_wikipedia_summary(query):
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wikipedia.run(query)

# Text field for user to input their GROQ API key
groq_api_key_input = st.text_input("Enter your GROQ API Key:")

# Columns layout for the button and text input

if st.button("Create a vector store"):
    if groq_api_key_input:
        groq_api_key = groq_api_key_input
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
        if uploaded_files:
            save_uploaded_files(uploaded_files)
            vector_embedding()
            st.write("Embeddings created!")
        else:
            st.write("Please upload at least one PDF file.")
    else:
        st.write("Please enter your GROQ API Key.")

# Prompt template
prompt = ChatPromptTemplate.from_template(
"""
Answer the question based on the provided context in at least 100 words. 
Provide the most accurate and detailed response based on the available information. 
Every correct response will get you a big prize.

## Based on PDF Content
<context>
{context}
</context>

## Based on Wikipedia Summary
<wikipedia_result>
{wikipedia_result}
</wikipedia_result>

## Based on Google Search Results
<google_result>
{google_result}
</google_result>

Question: {input}
"""
)

# User input for questions
prompt1 = st.text_input("Enter your question here:")

if prompt1:
    if groq_api_key_input:
        groq_api_key = groq_api_key_input
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Fetch Google and Wikipedia results    
        wikipedia_result = fetch_wikipedia_summary(prompt1)
        google_result = google_output(prompt1)
        # Update context with results
        context = {
            "google_result" : google_result,
            "wikipedia_result": wikipedia_result,
            "input": prompt1,
        }

        response = retrieval_chain.invoke(context)
        st.write(response["answer"])
    else:
        st.write("Please enter your GROQ API Key.")
