# Document Question Answering System (RAG)

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) system for answering questions from custom documents such as PDFs and TXT files. The application uses a Streamlit frontend, Cohere for embeddings and generation, and Pinecone for vector storage.

## Features
- Upload multiple PDF/TXT files
- Extract and clean text
- Split text into chunks with configurable size and overlap
- Generate embeddings via Cohere
- Build and update a Pinecone vector index
- Retrieve top-k relevant chunks
- Generate grounded answers using Cohere
- Generate execution logs and system metrics

## Architecture
```text
Document Upload
  -> Text Extraction
  -> Chunking
  -> Cohere Embedding Generation
  -> Pinecone Vector Store
  -> Query Embedding
  -> Similarity Search
  -> Cohere Answer Generation
```

## Installation
1. Create a Python 3.11+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file using `.env.example` and add your Cohere and Pinecone API keys.

## Running the App
```bash
streamlit run src/app.py
```

## Folder Structure
```text
project/
├── data/
├── database/
├── logs/
├── reports/
├── src/
│   ├── app.py
│   ├── chatbot.py
│   ├── vectorstore.py
│   ├── config.py
│   ├── logger.py
│   ├── metrics.py
│   └── utils.py
├── README.md
├── requirements.txt
└── .env.example
```

## Example Usage
1. Upload sample PDF/TXT files.
2. Click "Build Knowledge Base".
3. Ask a question such as "What is the main benefit of this document?"

## Screenshots
Placeholder for screenshots.

## Future Improvements
- Add DOCX support
- Add hybrid search and re-ranking
- Add conversation history
- Add source page and document metadata display
