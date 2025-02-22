import re
import asyncio

def detect_follow_up_question(query: str):
    """Check if a follow-up question is needed based on the user query."""
    query_lower = query.lower()

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


from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_community import GoogleSearchResults, GoogleSearchAPIWrapper
from rag_pipeline import rag_pipeline

# Define prompt template for structured responses
custom_prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template="""
You are NamasteBot, a multilingual AI assistant.
- Answer queries in the same language as the user.
- Retrieve knowledge from the **RAG pipeline**.
- Use **llm** for real-time information.
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

# search = GoogleSearchResults(api_wrapper=GoogleSearchAPIWrapper(
#     google_api_key=GOOGLE_API_KEY,
#     google_cse_id=GOOGLE_CSE_ID
# ))

def run_rag_pipeline(query):
    return rag_pipeline(query)  # Sync call

rag_tool = Tool(
    name="Multilingual RAG",
    func=run_rag_pipeline,
    description="Retrieves multilingual information from a knowledge base."
)

# Initialize LLM
llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=GOOGLE_API_KEY)

# Define synchronous custom agent
class SyncCustomAgent:
    def __init__(self, llm, tools, memory, prompt):
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.prompt = prompt
        self.pending_follow_up = None  # Track follow-up questions

    def invoke(self, query):
        """Process user query, handle follow-ups, and generate responses with RAG first."""
        print("\n> Entering new agentExecutor chain...")

        if self.pending_follow_up:
            full_query = f"Follow-up response: {query}"
            self.pending_follow_up = None
        else:
            follow_up_question = detect_follow_up_question(query)
            if follow_up_question:
                self.pending_follow_up = follow_up_question
                return follow_up_question  

            full_query = query

        print(f"Question: {query}\n")
    
        # Try using the RAG pipeline first
        print(f"Thought: Checking RAG for relevant knowledge...\n")
        rag_result = run_rag_pipeline(full_query)

        # List of generic or unhelpful RAG responses
        generic_rag_responses = [
            "This text focuses on Goa and Portuguese colonization",
            "I don’t have enough relevant information on this topic",
            "This text contains historical details but lacks a direct answer",
            "This text doesn't mention",
            "This text focuses on Goa,",
            "This text does not contain any information about "
        ]

        if rag_result and len(rag_result.strip()) > 5 and not any(phrase in rag_result for phrase in generic_rag_responses):
            response = rag_result
            print("RAG Response Found ✅")
        else:
            print("RAG result insufficient. Falling back to LLM ❌")
            # Format the prompt
            chat_history = self.memory.chat_memory.messages
            formatted_prompt = self.prompt.format(input=full_query, chat_history=chat_history)

            # Use the LLM for response generation
            response = self.llm.invoke(formatted_prompt)

        # Store conversation in memory
        self.memory.chat_memory.add_message(HumanMessage(content=query))
        self.memory.chat_memory.add_message(AIMessage(content=response))

        print(f"Final Answer: {response}\n")
        print("> Finished chain.\n")

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
