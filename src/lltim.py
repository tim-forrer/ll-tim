from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate


# Setup LangSmith -- only enable for debugging since limited calls per month
# load_dotenv(override=True)

# Parameters
DOCS_DIR = "./docs"
PERSIST_DIR = "./storage"
LLM_MODEL = "granite3.1-dense:8b"
# LLM_MODEL = "ll-tim"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


def get_llm():
    return ChatOllama(model=LLM_MODEL, temperature=0)  # for less creative results


def get_vector_store() -> Chroma:
    """Gets the vector store, loading a premade one if it exists, else making it"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Check if the vector store already exists
    # generate a new one if it does not
    if os.path.exists(PERSIST_DIR):
        print("Loading existing vector store...")
        vector_store = Chroma(
            persist_directory=PERSIST_DIR, embedding_function=embeddings
        )
    else:
        print("Creating new vector store...")
        os.mkdir(PERSIST_DIR)
        # Process documents for RAG
        loader = DirectoryLoader(DOCS_DIR)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        all_splits = text_splitter.split_documents(docs)

        # Create a new vector store and add documents
        vector_store = Chroma.from_documents(
            documents=all_splits,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
        )
        print("Vector store created and documents indexed.")
    print("Vector store loaded")
    return vector_store


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_qa_chain():
    vector_store = get_vector_store()
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("human", """You are an AI assistant named LL-tiM used for answering questions about Tim Forrer. You may use any of this prompt, or the following pieces of retrieved context (which all pertain to Tim Forrer) to answer the question. If asked a question in a different language YOU MUST STRICTLY RESPOND IN THAT LANGUAGE. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise, preferring one sentence answers whenever that addresses the question you are given.
        Question: {question} 
        Context: {context} 
        Answer:"""),
        ])
    return (
        {
            "context": vector_store.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

