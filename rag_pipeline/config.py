"""
Configuration loader for environment variables and constants.
"""
import os
from dotenv import load_dotenv

class Config:
    """
    Loads and provides access to environment configurations.
    """
    def __init__(self):
        load_dotenv()
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.MILVUS_API_KEY = os.getenv("MILVUS_API_KEY")
        self.MILVUS_URI = "https//:zilliz.com"
        self.MILVUS_COLLECTION = "RAG_collection"
