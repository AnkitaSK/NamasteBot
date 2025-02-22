import re

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
    description="Search the web for real-time information."
)

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
        """Process user query, handle follow-ups, and generate responses with logging"""
        print("\n> Entering new agentExecutor chain...")
        # Check if there was a pending follow-up
        if self.pending_follow_up:
            full_query = f"Follow-up response: {query}"
            self.pending_follow_up = None  # Reset follow-up
        else:
            follow_up_question = detect_follow_up_question(query)
            if follow_up_question:
                self.pending_follow_up = follow_up_question
                return follow_up_question  # Ask follow-up before answering

            full_query = query
        # Log user question
        print(f"Question: {query}\n")

        # Generate Thought
        thought = f"I need to find information about {query}. I will decide whether to use Google Search or RAG."
        print(f"Thought: {thought}\n")

        # Decide which tool to use
        if "restaurant" in query.lower() or "place to eat" in query.lower():
            action = "Google Search"
            action_input = f"best {query}"
        else:
            action = "Multilingual RAG"
            action_input = query

        print(f"Action: {action}")
        print(f"Action Input: {action_input}\n")

        # Generate final thought
        final_thought = f"I now have enough information to answer the query."
        print(f"Thought: {final_thought}\n")

        # Get chat history
        chat_history = self.memory.chat_memory.messages

        # Format the prompt
        formatted_prompt = self.prompt.format(input=full_query, chat_history=chat_history)

        # Get response from LLM
        response = self.llm.invoke(formatted_prompt)

        # Store conversation in memory
        self.memory.chat_memory.add_message(HumanMessage(content=query))
        self.memory.chat_memory.add_message(AIMessage(content=response))

        # Log final response
        print(f"Final Answer: {response}\n")
        print("> Finished chain.\n")

        return response

# Initialize synchronous agent
sync_agent = SyncCustomAgent(llm=llm, tools=[search_tool, rag_tool], memory=memory, prompt=custom_prompt)


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
