# RAG Application

This RAG (Retrieval-Augmented Generation) application allows users to input a query and upload a PDF file. The application searches the PDF, Wikipedia, and Google search engine to return a detailed summary about the input text.

## Features

- **User Input**: Accepts user queries.
- **PDF Upload**: Users can upload a PDF file for the application to search.
- **Information Retrieval**: Searches the provided PDF, Wikipedia, and Google for relevant information.
- **Summary Generation**: Generates a detailed summary based on the retrieved information.

## Application
https://github.com/user-attachments/assets/4da9c9fe-5de8-4753-9a81-8848ead754fb


## Requirements

- Python
- Streamlit
- python-dotenv
- wikipedia-api
- PyPDF2
- langchain-community-tools
- langchain-core
- groq

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/Rey8el/RAG-application.git
    cd RAG-application
    ```

2. **Create a virtual environment**:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # For macOS/Linux
    venv\Scripts\activate  # For Windows
    ```

3. **Install the dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the application**:

    ```bash
    streamlit run app.py
    ```

2. **Upload a PDF file** and enter your query in the provided input field.

3. **Retrieve and view the summary** generated based on the content of the PDF, Wikipedia, and Google search results.

## Example

1. **Input Query**: "What is the impact of climate change on polar bears?"

2. **Upload PDF**: Upload a PDF document containing information about climate change.

3. **Output**: The application will search the PDF, Wikipedia, and Google for information related to the query and provide a detailed summary.

## Project Structure

```plaintext
rag-application/
├── app.py
├── requirements.txt
├── README.md
└── .env.example
