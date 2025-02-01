import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing import List, TypedDict

# Load environment variables
load_dotenv()
LANGSMITH_API_KEY = os.getenv("LANGSMITH")
DOCS_DIR = "./docs"
PERSIST_DIR = "./storage"
LLM_MODEL = "llama3.2:1b"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Choose model + embedding
model = OllamaLLM(model=LLM_MODEL)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def create_graph() -> CompiledStateGraph:
    # Check if the vector store already exists
    if os.path.exists(PERSIST_DIR):
        print("Loading existing vector store...")
        vector_store = Chroma(
            persist_directory=PERSIST_DIR, embedding_function=embeddings
        )
    else:
        print("Creating new vector store...")
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

    # Define prompt for question-answering
    prompt = PromptTemplate.from_template(
        "Answer the question based on the following context, limiting your answers to one line:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    # Define application steps
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"], k=3)
        return {"context": retrieved_docs, "question": state["question"]}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.format(question=state["question"], context=docs_content)
        response = model.invoke(messages)
        return {"answer": response}

    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    return graph
