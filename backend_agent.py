from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_community import GoogleSearchResults
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType

# Import the RAG pipeline function from rag_pipeline.py
from rag_pipeline import rag_pipeline_with_translation

app = FastAPI()

# Initialize Google Gemini API
GOOGLE_API_KEY = 'AIzaSyCRgdG7aYZD74STvn9LJNC812LEgJT0a7A' #os.getenv("GOOGLE_API_KEY")
llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=GOOGLE_API_KEY)

# Google Search API Tool

# 
# search_tool = GoogleSearchResults(api_key='d5eebcc0392094387')
GOOGLE_CSE_ID = 'd5eebcc0392094387'
search = GoogleSearchResults(api_wrapper=GoogleSearchAPIWrapper(
    google_api_key=GOOGLE_API_KEY,
    google_cse_id=GOOGLE_CSE_ID
))

search_tool = Tool(
    name="Google Search",
    func=search.run,
    description="Search the web for relevant information."
)

def debug_rag_tool(query):
    print(f"Received query: {query}")  # Debugging output
    try:
        result = asyncio.get_event_loop().run_until_complete(rag_pipeline_with_translation(query))
        print(f"RAG result: {result}")  # Debugging output
        return result
    except Exception as e:
        print(f"Error in RAG pipeline: {str(e)}")  # Debugging output
        return f"Error: {str(e)}"
    
# Create LangChain Tool for RAG Pipeline
rag_tool = Tool(
    name="Multilingual RAG",
    func=debug_rag_tool,
    description="Retrieves multilingual information from a knowledge base."
)

# Initialize Agent with Tools
tools = [rag_tool]
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    try:
        response = await asyncio.to_thread(agent.run, query.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

