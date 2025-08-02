"""
Class to load PDF documents.
"""
from langchain_community.document_loaders import PyPDFLoader

class PDFLoader:
    """
    Loads and parses PDF files into documents.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        """
        Loads the PDF and returns documents.
        Returns:
            list: Loaded documents.
        """
        loader = PyPDFLoader(self.file_path)
        return loader.load()
