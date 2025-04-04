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

import os
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
import logging
from pydantic import BaseModel, Field
from typing import Optional, Annotated, Sequence, Literal
from typing_extensions import TypedDict
import custom_prompts  # type: ignore


class Settings:
    LLM_MODEL = "gemma3:4b"
    TOOL_MODEL = "llama3.2:latest"
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    DOCS_DIR = "./docs"
    PERSIST_DIR = "./persist"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    # The list of messages is the state that is passed through the graph
    user_query: str
    rag_query: Optional[str]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: Optional[list[Document]]
    answer: Optional[str]


class BinaryGrader(BaseModel):
    """Always use this to structure your response: either 'yes' or 'no'."""

    grade: Literal["yes", "no"] = Field(description="Your response, 'yes' or 'no'.")


class RAGGraph:
    def __init__(self):
        logger = logging.getLogger(__name__)
        base_llm = ChatOllama(
            model=Settings.LLM_MODEL,
            temperature=0.2,
        )
        tool_llm = ChatOllama(
            model=Settings.TOOL_MODEL,
            temperature=0.2,
        )

        # ===Vector Store===
        embeddings = HuggingFaceEmbeddings(model_name=Settings.EMBEDDING_MODEL)
        loader = DirectoryLoader(Settings.DOCS_DIR)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Settings.CHUNK_SIZE, chunk_overlap=Settings.CHUNK_OVERLAP
        )
        docs = loader.load()
        all_splits = text_splitter.split_documents(docs)

        if os.path.exists(Settings.PERSIST_DIR):
            # Load existing persisted vector store
            logger.info("Loading existing vector store...")
            vector_store = Chroma(
                embedding_function=embeddings, persist_directory=Settings.PERSIST_DIR
            )
        else:
            # Create new vector store and persist
            logger.info("No vector store found, building new one...")
            vector_store = Chroma.from_documents(
                documents=all_splits,
                embedding=embeddings,
                persist_directory=Settings.PERSIST_DIR,
            )
        logger.info("Vector store loaded!")

        # ===Conditional Edges===
        def related_to_tim(state: AgentState) -> Literal["generate", "retrieve"]:
            """Decide whether the user query is relevant to Tim Forrer or not."""
            logger.debug("Deciding relevance...")

            prompt = custom_prompts.related_to_tim

            relevance_grader = prompt | tool_llm.with_structured_output(BinaryGrader)
            relevance_grade = relevance_grader.invoke(
                {"user_query": state["user_query"], "messages": state["messages"]}
            )

            logger.debug(
                f"LLM thinks the query is relevant to Tim Forrer? {relevance_grade.grade}"
            )

            if relevance_grade.grade == "no":
                logger.debug("Generating answer directly")
                return "generate"
            else:
                logger.debug("Retrieving additional documents")
                return "retrieve"

        def document_grader(state: AgentState) -> Literal["generate", "rewrite"]:
            """Decide whether retrieved documents are relevant to the user query or not."""
            logger.debug("Grading documents...")

            prompt = custom_prompts.document_grader
            document_grader = prompt | tool_llm.with_structured_output(BinaryGrader)
            document_grade = document_grader.invoke(
                {"user_query": state["user_query"], "context": state["context"]}
            )

            logger.debug(
                f"LLM thinks retrieved documents are relevant? {document_grade.grade}"
            )

            if document_grade.grade == "yes":
                logger.debug("Generating answer using context")
                return "generate"
            else:
                logger.debug("Re-writing the user query for better RAG optimality")
                return "rewrite"

        # ===Nodes===
        def retrieve(state: AgentState) -> dict[str, str]:
            """
            Query the vector database for relevant documents to the user query.

            Uses the rewritten query if it is available, else use the user query.
            """
            logger.debug("Retrieving documents...")

            retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            if state["rag_query"] is None:
                state["rag_query"] = state["user_query"]
            docs = retriever.invoke(state["rag_query"])

            logger.debug("Retrieved documents:")
            if logger.isEnabledFor(logging.DEBUG):
                for doc in docs:
                    logger.debug(doc.page_content)
            # TODO: Format the documents more nicely
            return {"context": docs}

        def rewrite(state: AgentState) -> dict[str, str]:
            """Rewrite the user query so that it is more appropriate for RAG."""
            logger.debug("Re-writing user query...")
            prompt = custom_prompts.rewriter
            rewriter = prompt | base_llm

            rewritten_query = rewriter.invoke({"user_query": state["user_query"]})
            logger.debug(f"Rewritten query: {rewritten_query.content}")
            return {"rag_query": rewritten_query.content}

        def generate(state: AgentState) -> dict[str, str]:
            """Generate an answer to the user query."""
            logger.debug("Generating answer...")
            prompt = custom_prompts.generator
            generator = prompt | base_llm
            response = generator.invoke(
                {"question": state["user_query"], "context": state["context"]}
            )
            logger.debug(f"Generated answer: {response.content}")
            return {"messages": response, "answer": response.content}

        # ===Graph===
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

    def query(self, user_query: str, uuid: int) -> str:
        logger = logging.getLogger(__name__)
        logger.debug(f"Querying graph with query {user_query}, UUID: {uuid}")
        messages = [
            HumanMessage(user_query)
        ]  # TODO: Some logic that loads a message history from a database.
        input = AgentState(
            {
                "user_query": user_query,
                "messages": messages,
                "rag_query": None,
                "context": None,
                "answer": None,
            }
        )
        logger.debug(f"Input to graph:\n{input}")
        response = self.graph.invoke(input=input)
        logger.debug(f"LLM final response:\n{response["answer"]}")
        if response["answer"] is None:
            raise RuntimeError("No response was given by the LLM.")
        return response["answer"]
