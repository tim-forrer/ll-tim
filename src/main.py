import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from lltim import RAGGraph  # type: ignore
from logging.config import dictConfig

ORGINS = ["https://tim-forrer.vercel.app", "http://localhost:3000"]

# Custom logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["console"],
            "level": "WARNING",
        },
        "uvicorn": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "lltim": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}
dictConfig(LOGGING_CONFIG)


class QueryRequest(BaseModel):
    query: str
    uuid: int


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORGINS,
    allow_credentials=True,
    allow_methods=["POST"],  # Allow all HTTP methods (e.g., GET, POST, OPTIONS)
    allow_headers=["*"],  # Allow all headers (e.g., Content-Type, Authorization)
)

rag_graph = RAGGraph()


@app.post("/query")
async def query(request: QueryRequest):
    try:
        response = rag_graph.query(request.query, request.uuid)
        return {"response": response}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy"}


# Run the API
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001, log_config=LOGGING_CONFIG)
