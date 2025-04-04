"""
Basic idea:
Use LangGraph to create an AI chatbot that can answer questions about myself.

Basic pipeline:
    1. Recieve query from user.
    2. LLM first decides whether a query is about Tim Forrer or not.
    3. If yes, use a retriever tool to get the necessary context needed to answer the question.
    4. If no, come up with an answer to the question.
        - Prompt allows the LLM to say that it doesn't know.
        - If it doesn't know, use a web query to retrieve additional context.
    5. Generate a final answer.
    6. Output this answer.

References
https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/#retriever
"""

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel, Field
from typing import Optional, Annotated, Sequence, Literal
from typing_extensions import TypedDict
import custom_prompts  # type: ignore


class Settings:
    LLM_MODEL = "llama3.2"
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    DOCS_DIR = "./docs"


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    # The list of messages is the state that is passed through the graph
    user_query: str
    rag_query: Optional[str]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: list[Document]
    answer: str


class BinaryGrader(BaseModel):
    """Always use this to structure your response: either 'yes' or 'no'."""

    grade: Literal["yes", "no"] = Field(description="Your response, 'yes' or 'no'.")


class RAGGraph:
    def __init__(self):
        base_llm = ChatOllama(
            model=Settings.LLM_MODEL,
            temperature=0,
        )
        embeddings = HuggingFaceEmbeddings(model_name=Settings.EMBEDDING_MODEL)

        loader = DirectoryLoader(Settings.DOCS_DIR)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )

        docs = loader.load()
        all_splits = text_splitter.split_documents(docs)
        vector_store = FAISS.from_documents(
            documents=all_splits,
            embedding=embeddings,
        )

        def related_to_tim(state: AgentState) -> Literal["generate", "retrieve"]:
            """Decide whether the user query is relevant to Tim Forrer or not."""
            prompt = custom_prompts.related_to_tim

            relevance_grader = prompt | base_llm.with_structured_output(BinaryGrader)
            relevance_grade = relevance_grader.invoke(
                {"user_query": state["user_query"], "messages": state["messages"]}
            )

            if relevance_grade.grade == "no":
                return "generate"
            else:
                return "retrieve"

        def retrieve(state: AgentState) -> dict[str, str]:
            """
            Query the vector database for relevant documents to the user query.

            Uses the rewritten query if it is available, else use the user query.
            """
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            if state["rag_query"] is None:
                state["rag_query"] = state["user_query"]
            docs = retriever.invoke(state["rag_query"])

            return {"context": docs}

        def document_grader(state: AgentState) -> Literal["generate", "rewrite"]:
            """Decide whether retrieved documents are relevant to the user query or not."""
            prompt = custom_prompts.document_grader
            document_grader = prompt | base_llm.with_structured_output(BinaryGrader)
            document_grade = document_grader.invoke(
                {"user_query": state["user_query"], "context": state["context"]}
            )

            if document_grade.grade == "yes":
                return "generate"
            else:
                return "rewrite"

        def rewrite(state: AgentState) -> dict[str, str]:
            """Rewrite the user query so that it is more appropriate for RAG."""
            prompt = custom_prompts.rewriter
            rewriter = prompt | base_llm

            rewritten_query = rewriter.invoke({"user_query": state["user_query"]})
            return {"rag_query": rewritten_query.content}

        def generate(state: AgentState) -> dict[str, str]:
            """Generate an answer to the user query."""
            prompt = custom_prompts.generator
            generator = prompt | base_llm
            response = generator.invoke(
                {"question": state["user_query"], "context": state["context"]}
            )

            return {"messages": response, "answer": response.content}

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("rewrite", rewrite)
        workflow.add_node("generate", generate)

        # Add edges
        workflow.add_conditional_edges(START, related_to_tim)
        workflow.add_conditional_edges("retrieve", document_grader)
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("generate", END)

        self.graph = workflow.compile()
