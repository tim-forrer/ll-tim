import os.path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

BASE_PROMPT = """
You are an expert on all things Tim Forrer.
Your function is to answer questions about Tim, posed by users of the website.
Keep your responses brief where possible. Two sentences should usually suffice, but more is acceptable where needed.
In your response, only provide information if it is directly related to the user input.
Prior knowledge can be incorporated into your answer if it is somewhat relevant to Tim Forrer.
Feel free to write using your own words, without directly copying the phrasing in the documents you've been given.
What follows after the '===' is the question from the user.\n
===\n
"""


class QueryRequest(BaseModel):
    query: str


# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
# ollama
Settings.llm = Ollama(model="granite3.1-dense:8b", request_timeout=360.0)

PERSIST_DIR = "./storage"
print("Loading documents...")
if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader("./docs").load_data()
    index = VectorStoreIndex.from_documents(
        documents,
    )
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()

# API endpoint
orgins = ["https://tim-forrer.vercel.app", "http://localhost:3000"]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=orgins,
    allow_credentials=True,
    allow_methods=["POST"],  # Allow all HTTP methods (e.g., GET, POST, OPTIONS)
    allow_headers=["*"],  # Allow all headers (e.g., Content-Type, Authorization)
)


@app.post("/query")
async def query(request: QueryRequest):
    try:
        query = BASE_PROMPT + request.query
        response = query_engine.query(query)
        return {"response": str(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy"}


# Run the API
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
