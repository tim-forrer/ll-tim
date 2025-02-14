import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from lltim import create_graph  # type: ignore


class QueryRequest(BaseModel):
    query: str


ORGINS = ["https://tim-forrer.vercel.app", "http://localhost:3000"]


graph = create_graph()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORGINS,
    allow_credentials=True,
    allow_methods=["POST"],  # Allow all HTTP methods (e.g., GET, POST, OPTIONS)
    allow_headers=["*"],  # Allow all headers (e.g., Content-Type, Authorization)
)


@app.post("/query")
async def query(request: QueryRequest):
    try:
        messages = graph.invoke(
            {"messages": [{"role": "user", "content": request.query}]}
        )
        response = messages["messages"][-1].content
        return {"response": response}
    except Exception as e:
        print("Error occured:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy"}


# Run the API
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
