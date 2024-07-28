import streamlit as st
from dotenv import load_dotenv
import os
import wikipediaapi
import PyPDF2 as pdf
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, GoogleSearchAPIWrapper
from langchain_core.tools import Tool
import json
from groq import Groq 

# Load environment variables from .env file
load_dotenv()

google_api_key = os.environ.get("GOOGLE_SEARCH_API_KEY")
google_cse_id = os.environ.get("GOOGLE_CSE_ID")
groq_api_key = os.environ.get("GROQ_API_KEY")

if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
    client = Groq(api_key=groq_api_key)  
else:
    st.error("GROQ_API_KEY environment variable is not set.")
    st.stop()

def google_output(user_input):
    search = GoogleSearchAPIWrapper()
    tool = Tool(
        name="google_search",
        description="Search Google for recent results.",
        func=search.run,
    )
    google_result = tool.run(user_input)
    return google_result

def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

def fetch_wikipedia_summary(query):
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wikipedia.run(query)

def main():
    st.title("RAG Application")
    user_input = st.text_input("Enter your question:")
    uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Please upload the PDF")

    if st.button("Submit"):
        if not user_input:
            st.error("Please enter a question.")
            st.stop()
        
        if not uploaded_file:
            st.error("Please upload a PDF file.")
            st.stop()
        
        try:
            text = input_pdf_text(uploaded_file)
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
            st.stop()

        if len(text) > 10000:  # Truncate text if too long
            text = text[:10000] + "..."

        try:
            wiki_summary = fetch_wikipedia_summary(user_input)
        except Exception as e:
            st.error(f"Error fetching Wikipedia summary: {e}")
            st.stop()
        
        brave_summary = google_output(user_input)

        messages = [
            {
                "role": "system",
                "content": """
                You are a helpful AI chatbot which responds to user by searching the user input in the text provided to you via PDF and Wikipedia Summary and Google Search.
                Provide your answer in MD format with 3 headings: one for the response based on the PDF, another for the response based on 
                the Wikipedia summary, and the last one based on the Google search.
                Each response in itself must contain at least 250 words and detailed explaination about the user input based on the source.
                This is the most important step, In the end using PDF, Wikipedia summary and Google Search generate a detailed summary of atleast 500 words on the topic.
                For every correct answer, you get $1000.
                """
            },
            {
                "role": "user",
                "content": f"user input = {user_input}, PDF = {text}, Wikipedia Summary = {wiki_summary}, Google Search = {brave_summary}"
            }
        ]

        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model="llama3-8b-8192",
            )

            response = chat_completion.choices[0].message.content
            st.markdown(response)
        except Exception as e:
            st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()