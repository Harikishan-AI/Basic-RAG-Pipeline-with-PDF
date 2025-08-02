"""
Vectorstore modules for Milvus and metadata cleaning.
"""
from langchain_community.vectorstores import Milvus
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def clean_metadata(metadata: dict) -> dict:
    """
    Cleans metadata keys for Milvus compatibility.
    Args:
        metadata (dict): Original metadata.
    Returns:
        dict: Cleaned metadata.
    """
    return {k.replace(".", "_").replace("-", "_"): v for k, v in metadata.items()}

class MilvusVectorStore:
    """
    Manages Milvus vector store creation and operations.
    """
    def __init__(self, config):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.connection_args = {
            "uri": config.MILVUS_URI,
            "token": config.MILVUS_API_KEY,
            "collection_name": config.MILVUS_COLLECTION
        }
        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            connection_args=self.connection_args,
            auto_id=True,
            index_params={
                "index_type": "HNSW",
                "metric_type": "L2"
            }
        )

    def add_documents(self, documents):
        """
        Embeds and adds split documents to the vector store.
        Args:
            documents (list): List of split document chunks.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [clean_metadata(doc.metadata) for doc in documents]
        self.vector_store.add_texts(texts=texts, metadatas=metadatas)

    def get_retriever(self, k=5):
        """
        Returns a retriever for semantic search.
        Args:
            k (int): Number of results to retrieve.
        Returns:
            retriever: A retriever instance.
        """
        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k},
        )
