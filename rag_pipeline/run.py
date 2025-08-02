"""
Main script to interact with the RAG pipeline.
"""
from config import Config
from pdf_loader import PDFLoader
from splitter import DocumentSplitter
from vectorstore import MilvusVectorStore
from rag import RAGChain

def main():
    """
    Runs the RAG QA loop.
    """
    # Load configuration
    config = Config()
    
    # Step 1: Load PDF
    pdf_loader = PDFLoader("1706.03762v7.pdf")
    documents = pdf_loader.load()
    print("✅ PDF loaded.")

    # Step 2: Split Documents
    splitter = DocumentSplitter(chunk_size=1000, chunk_overlap=200)
    splitted_documents = splitter.split(documents)
    print("✅ Documents split.")

    # Step 3: Embed and Upload to Milvus
    vector_db = MilvusVectorStore(config)
    vector_db.add_documents(splitted_documents)
    print("✅ Documents embedded and stored in Milvus.")

    # Step 4: Create Retriever and RAG Chain
    retriever = vector_db.get_retriever(k=5)
    rag_chain = RAGChain(retriever)

    print("\n💡 Ready! Ask questions about your PDF (type 'exit' to quit).\n")

    while True:
        query = input("🧠 Your Question: ")
        if query.lower() in ("exit", "quit"):
            print("👋 Exiting.")
            break
        answer = rag_chain.ask(query)
        print("🤖 Gemini Answer:\n", answer, "\n")

if __name__ == "__main__":
    main()
