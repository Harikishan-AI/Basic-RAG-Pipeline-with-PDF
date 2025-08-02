"""
RAG chain module for retrieval-augmented response.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class RAGChain:
    """
    Sets up and manages the Retrieval-Augmented Generation (RAG) chain.
    """
    def __init__(self, retriever):
        self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.prompt = PromptTemplate(
            template=(
                "You are a helpful assistant. Use the following pieces of context to answer the question at the end.\n"
                "{context}\n"
                "Question: {question}\n"
                "Answer:"
            ),
            input_variables=["context", "question"]
        )
        self.chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.model
            | StrOutputParser()
        )

    def ask(self, query: str) -> str:
        """
        Answers a user query using the RAG chain.
        Args:
            query (str): The input question.
        Returns:
            str: Model-generated answer.
        """
        return self.chain.invoke(query)
