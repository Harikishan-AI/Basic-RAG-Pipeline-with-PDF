from setuptools import setup, find_packages

setup(
    name="rag_pipeline",
    version="0.1.0",
    description="A modular RAG pipeline for PDF question answering with Milvus and Gemini",
    author="Harikishan",
    packages=find_packages(),
    install_requires=[
        "python-dotenv",
        "langchain-community",
        "langchain-google-genai",
        "langchain-core",
        "pypdf",
        "nltk",
        "pymilvus",
    ],
    python_requires=">=3.11",
)
