import re
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain_google_community import GoogleSearchResults, GoogleSearchAPIWrapper
from rag_pipeline import rag_pipeline

class FollowUpTracker:
    def __init__(self):
        self.follow_up_count = 0
        self.max_follow_ups = 2

    def reset(self):
        self.follow_up_count = 0

    def can_ask_follow_up(self):
        return self.follow_up_count < self.max_follow_ups

    def increment(self):
        self.follow_up_count += 1

def detect_follow_up_question(query: str):
    query_lower = query.lower()

    # Immediate response for temperature-related queries
    if re.search(r"temperature|weather|climate", query_lower):
        return "Please check Google Search for the latest temperature information."

    follow_up_questions = {
        r"eat|restaurant|food|dining": "Do you prefer vegetarian or non-vegetarian food?",
        r"hotel|stay|accommodation": "Are you looking for budget or luxury options?",
        r"tourist spot|places to visit|sightseeing": "Do you prefer historical sites or natural attractions?",
        r"flight|travel|trip": "Do you need help with flight bookings or itinerary planning?"
    }

    for pattern, follow_up in follow_up_questions.items():
        if re.search(pattern, query_lower):
            return follow_up

    return None  # No follow-up required    

# Define prompt template for structured responses
custom_prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template="""
You are a multilingual AI assistant.
- Answer queries in the same language as the user.
- Retrieve knowledge from the **RAG pipeline**.
- Use **Google Search** for real-time information.
- If a follow-up question is required, ask it before answering.

### **Chat History:**  
{chat_history}

### **User Query:**  
{input}

### **Your Response:**  
"""
)

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory.chat_memory.add_message(SystemMessage(content="You are a helpful assistant."))

# Define tools (RAG & Google Search)
GOOGLE_API_KEY = "AIzaSyBNu0Ea8aWQ1JnvXDOqSsnmH8Q2xZS7qww"
GOOGLE_CSE_ID = "f481134eb108c4222"

search = GoogleSearchResults(api_wrapper=GoogleSearchAPIWrapper(
    google_api_key=GOOGLE_API_KEY,
    google_cse_id=GOOGLE_CSE_ID
))

search_tool = Tool(
    name="Google Search",
    func=search.run,
    description="Search the web for real-time information."
)

def run_rag_pipeline(query):
    return rag_pipeline(query)  # Sync call

rag_tool = Tool(
    name="Multilingual RAG",
    func=run_rag_pipeline,
    description="Retrieves multilingual information from a knowledge base."
)

# Define synchronous custom agent
import asyncio  # Import asyncio for running async functions

class SyncCustomAgent:
    def __init__(self, tools, memory, prompt):
        self.tools = tools
        self.memory = memory
        self.prompt = prompt
        self.pending_follow_up = None  # Track follow-up questions
        self.follow_up_tracker = FollowUpTracker()  # Initialize follow-up tracker

    def invoke(self, query):
        try:
            print("\n> Entering new agentExecutor chain...")
            print(f"Question: {query}\n")

            if re.search(r"temperature|weather|climate", query.lower()):
                return "Please check Google Search for the latest temperature information."

            if self.pending_follow_up:
                full_query = f"Follow-up response: {query}"
                self.pending_follow_up = None
            else:
                if self.follow_up_tracker.can_ask_follow_up():
                    follow_up_question = detect_follow_up_question(query)
                    if follow_up_question:
                        self.follow_up_tracker.increment()
                        self.pending_follow_up = follow_up_question
                        return follow_up_question
                full_query = query

            print(f"Processing Query: {full_query}")

            if "restaurant" in query.lower() or "place to eat" in query.lower():
                response = search_tool.run(f"best {query}")
            else:
                response = asyncio.run(rag_pipeline(query))  # ✅ Ensure async function is awaited properly

            self.memory.chat_memory.add_message(HumanMessage(content=query))
            self.memory.chat_memory.add_message(AIMessage(content=response))

            print(f"Final Answer: {response}\n")
            print("> Finished chain.\n")

            self.follow_up_tracker.reset()
            return response    

        except Exception as e:
            print(f"Error in invoke(): {e}")  # Print error in the console
            return "An error occurred while processing your request."
 

# Initialize synchronous agent
sync_agent = SyncCustomAgent(tools=[search_tool, rag_tool], memory=memory, prompt=custom_prompt)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    try:
        # Invoke synchronous agent
        response = sync_agent.invoke(query.question)

        # Ensure response formatting
        formatted_response = response.strip()
        if formatted_response and not formatted_response.endswith(('.', '!', '?')):
            formatted_response += '.'

        return {"response": formatted_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    
