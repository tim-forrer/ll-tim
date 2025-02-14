import os
import dotenv
from pprint import pp
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, ToolMessage, BaseMessage
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.graph.state import CompiledStateGraph

# Setup LangSmith
dotenv.load_dotenv()

# Model parameters
DOCS_DIR = "./docs"
PERSIST_DIR = "./storage"
LLM_MODEL = "ll-time:latest"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Load LLM + embedding model
llm = ChatOllama(model=LLM_MODEL)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vector_store() -> Chroma:
    """Gets the vector store, loading a premade one if it exists, else making it"""
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
    return vector_store


def create_graph() -> CompiledStateGraph:
    vector_store = get_vector_store()

    @tool(response_format="content_and_artifact")
    def retrieve(query: str) -> tuple[str, list[Document]]:
        """Retrieve information for a query."""
        retrieved_docs = vector_store.similarity_search(query, k=3)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata} \n Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    def query_or_respond(state: MessagesState) -> dict[str, list[BaseMessage]]:
        """Tool call for retrieval or direct response"""
        llm_with_tools = llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    tools = ToolNode([retrieve])  # executes tool, adds result as ToolMessage to state

    def generate(state: MessagesState) -> dict[str, list[BaseMessage]]:
        """Generate an answer"""
        # Get all the most recently added messages
        # which are also ToolMessage
        # stopping once there are no more ToolMessages
        recent_tool_messages: list[ToolMessage] = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break

        tool_messages = recent_tool_messages[::-1]  # put messages in order
        docs_content = "\n\n".join(str(doc.content) for doc in tool_messages)
        system_message_content = f"""
            You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer
            the question. If you don't know the answer, say that you don't know.
            {docs_content}
            """
        # retrieve all the messages that are part of the actual conversation
        # i.e. not any toolcall messages
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in {"human", "system"}
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        response = llm.invoke(prompt)
        return {"messages": [response]}

    # Compile application and test
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond", tools_condition, {END: END, "tools": "tools"}
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)
    graph = graph_builder.compile()
    return graph


graph = create_graph()
response = graph.invoke({"messages": [{"role": "user", "content": "Hello"}]})
pp(response)

response = graph.invoke(
    {"messages": [{"role": "user", "content": "Who is Tim Forrer"}]}
)
pp(response)
