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
                    "You are an AI chatbot designed to provide comprehensive responses based on information extracted from a PDF document, Wikipedia, and Google search results. For each user query, follow these steps and structure your response using the specified format:

                    ### Instructions:

                    1. **PDF Information**:
                        - Extract relevant information from the provided PDF document.
                        - Write at least 250 words summarizing the key points related to the user query.

                    2. **Wikipedia Summary**:
                        - Search for a summary on Wikipedia related to the user query.
                        - Write at least 250 words summarizing the Wikipedia entry and how it relates to the query.

                    3. **Google Search**:
                        - Conduct a Google search for the user query.
                        - Write at least 250 words summarizing the top search results and provide additional insights or context.

                    4. **Summary**:
                        - Using the information from the PDF, Wikipedia, and Google search, write a detailed summary of at least 250 words.
                        - Integrate the key points from each source to provide a comprehensive answer to the user query.

                    ### Formatting:
                    - Use Markdown format for your response.
                    - Include three headings for each section as shown below:
                    1. ### Information from PDF
                    2. ### Wikipedia Summary
                    3. ### Google Search
                    4. ### Summary

                    ### Example Structure:

                    ```markdown
                    ### Information from PDF
                    [Provide information based on the PDF, at least 250 words]

                    ### Wikipedia Summary
                    [Provide information based on the Wikipedia summary, at least 250 words]

                    ### Google Search
                    [Provide information based on Google search results, at least 250 words]

                    ### Summary
                    [Provide a detailed summary based on all sources, at least 250 words]

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