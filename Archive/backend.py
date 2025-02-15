from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio

# Import the RAG pipeline function from rag_pipeline.py
from rag_pipeline import rag_pipeline_with_translation

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    try:
        response = await rag_pipeline_with_translation(query.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

