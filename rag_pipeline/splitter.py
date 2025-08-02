"""
Class for splitting documents into text chunks using NLTK.
"""
from langchain_text_splitters import NLTKTextSplitter

class DocumentSplitter:
    """
    Splits loaded documents into chunks for embedding.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = NLTKTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split(self, documents):
        """
        Splits documents.
        Args:
            documents (list): List of document objects.
        Returns:
            list: List of split document chunks.
        """
        return self.splitter.split_documents(documents)
