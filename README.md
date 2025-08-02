## Basic RAG Pipeline

A modular Retrieval-Augmented Generation (RAG) pipeline for question answering over PDF documents. 
Combines Milvus for vector storage and Google's Gemini for language generation, all wrapped in a clean, object-oriented Python package.

rag_pipeline/
│
├── __init__.py
├── config.py
├── pdf_loader.py
├── splitter.py
├── vectorstore.py
├── rag.py
├── run.py
├── ...
│
├── README.md
├── setup.py
└── .env                # Place your environment variables here

