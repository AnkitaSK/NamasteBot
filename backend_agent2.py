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

def run_rag_pipeline(query):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(rag_pipeline_with_translation(query))
    
# Create LangChain Tool for RAG Pipeline
rag_tool = Tool(
    name="Multilingual RAG",
    func=run_rag_pipeline,
    description="Retrieves multilingual information from a knowledge base."
)

# Initialize Agent with Tools
tools = [search_tool,rag_tool]
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# follow up
from langchain.tools import Tool
import re

# Follow-up logic based on detected patterns
def follow_up_logic(query):
    query_lower = query.lower()

    # Define follow-up questions based on query intent
    follow_up_questions = {
        "eat|restaurant|food|dining": "Do you prefer vegetarian or non-vegetarian food?",
        "hotel|stay|accommodation": "Are you looking for budget or luxury options?",
        "tourist spot|places to visit|sightseeing": "Do you prefer historical sites or natural attractions?"
    }

    # Check if a follow-up is needed
    for pattern, follow_up_question in follow_up_questions.items():
        if re.search(pattern, query_lower):
            return follow_up_question

    return None  # No follow-up needed


# LangChain Tool for Follow-Up Questions
follow_up_tool = Tool(
    name="Follow-up Question Generator",
    func=follow_up_logic,
    description="Analyzes queries and suggests follow-up questions when necessary."
)

# Add follow-up tool to the agent's tools list
tools.append(follow_up_tool)


try:
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )
except Exception as e:
    print("Error initializing agent:", e)

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    try:
        # Check if a follow-up question is needed
        follow_up = follow_up_logic(query.question)
        
        if follow_up:
            return {"response": follow_up}  # Ask the follow-up question first

        # Otherwise, proceed with normal RAG processing
        response = await asyncio.to_thread(agent.invoke, {"input": query.question})

        # Format response
        formatted_response = response.get("output", "").strip()
        if formatted_response and not formatted_response.endswith(('.', '!', '?')):
            formatted_response += '.'

        return {"response": formatted_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
