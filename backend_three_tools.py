import re

from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_community import GoogleSearchResults, GoogleSearchAPIWrapper
from rag_pipeline import rag_pipeline  # Assuming this is your RAG pipeline function
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio

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

    # Immediate response for temperature queries
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

# Define prompt template (modified to emphasize RAG)
custom_prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template="""
You are a multilingual AI assistant.
- Answer queries in the same language as the user.
- **Prioritize retrieving knowledge from the RAG pipeline.** Use Google Search only for real-time information or if the RAG pipeline doesn't have relevant information.
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

# Add system message to memory
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
    description="Search the web for real-time information. Use this ONLY if the RAG pipeline doesn't have the answer."
)

def run_rag_pipeline(query):
    return rag_pipeline(query)  # Sync call

rag_tool = Tool(
    name="Multilingual RAG",
    func=run_rag_pipeline,
    description="Retrieves multilingual information from a knowledge base.  **Prioritize using this tool.**"
)

# Initialize LLM (still needed for prompt formatting and follow-up questions)
llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=GOOGLE_API_KEY)


class SyncCustomAgent:
    def __init__(self, llm, tools, memory, prompt):
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.prompt = prompt
        self.pending_follow_up = None  # Track follow-up questions
        self.follow_up_tracker = FollowUpTracker()  # Initialize follow-up tracker

    def invoke(self, query):
        print("\n> Entering new agentExecutor chain...")

        # Check if the query is about temperature and return a direct response
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

        print(f"Question: {query}\n")
        print(f"Thought: I need to find information about {query}. I will **first try the RAG pipeline**.\n")

        action = "Multilingual RAG"  # Default to RAG
        action_input = query

        # Example: Check if real-time info is needed (customize as needed)
        if re.search(r"temperature|weather|stock price", query.lower()):
            action = "Google Search"
            action_input = query  # Or a more specific search query

        print(f"Action: {action}")
        print(f"Action Input: {action_input}\n")
        print(f"Thought: I now have enough information to answer the query.\n")

        chat_history = self.memory.chat_memory.messages
        # formatted_prompt = self.prompt.format(input=full_query, chat_history=chat_history)

        # Use the appropriate tool:
        if action == "Multilingual RAG":
            rag_result = run_rag_pipeline(action_input)
            response = rag_result
            # Check if RAG result is empty or insufficient. If so, fallback to Google Search
            if not response or len(response.strip()) < 5: # Customize "insufficient" criteria
                print("RAG result insufficient. Falling back to Google Search.")
                action = "Google Search"
                action_input = query
                search_result = search.run(action_input)
                response = search_result
        elif action == "Google Search":
            search_result = search.run(action_input)
            response = search_result
        else:
            response = "Error: Invalid action."


        self.memory.chat_memory.add_message(HumanMessage(content=query))
        self.memory.chat_memory.add_message(AIMessage(content=response))

        print(f"Final Answer: {response}\n")
        print("> Finished chain.\n")

        self.follow_up_tracker.reset()
        return response


# Initialize synchronous agent
sync_agent = SyncCustomAgent(llm=llm, tools=[rag_tool], memory=memory, prompt=custom_prompt)


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

    except ValueError as e:  # Example: Catch a ValueError
        raise HTTPException(status_code=500, detail=f"ValueError: {e}")
    except TypeError as e:   # Example: Catch a TypeError
        raise HTTPException(status_code=500, detail=f"TypeError: {e}")
    except Exception as e: # Keep this as a last resort
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")    
