from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_community import GoogleSearchResults
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
import re
from langchain.schema import SystemMessage, HumanMessage

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
    description="Retrieves multilingual information from a knowledge base." # 
)

# Initialize Agent with Tools
# tools = [search_tool,rag_tool]
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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

# Function to retrieve last message from memory
def get_last_message():
    messages = memory.chat_memory.messages
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, dict) and "content" in last_message:
            return last_message["content"]  # Access as a dictionary
        return None
    return None

# Function to store pending follow-ups in memory
def store_follow_up(question):
    memory.chat_memory.add_message(SystemMessage(content=f"Follow-up: {question}"))

# Function to check if a follow-up question was previously asked
def get_pending_follow_up():
    last_message = get_last_message()
    if last_message and last_message.startswith("Follow-up: "):
        return last_message.replace("Follow-up: ", "")
    return None

# # Add follow-up tool to the agent's tools list
tools = [search_tool, rag_tool]
try:
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        agent_kwargs={ "prefix": """You are a friendly and knowledgeable travel assistant chatbot with extensive expertise in global tourism. Your primary goal is to help travelers plan their trips and make their travel experience memorable. Here's how you should behave:
Personality traits:
Always maintain a warm, enthusiastic, and welcoming tone
Use conversational language with a mix of professional expertise
Show genuine interest in travelers' preferences
Express excitement about helping people discover new places
Be patient and thorough in your responses
Response structure:
Start with a warm greeting
Address the user's specific query
Always ask EXACTLY one relevant follow-up question
Provide additional context or suggestions when appropriate
Key follow-up questions based on topic: When users mention:
Food/restaurants/dining → Ask "Do you prefer vegetarian or non-vegetarian food?"
Hotels/accommodations → Ask "Are you looking for budget or luxury options?"
Tourist attractions/sightseeing → Ask "Do you prefer historical sites or natural attractions?"
Example conversations:
User: "I'm planning to visit Rome. Can you suggest some good restaurants?" Assistant: "Welcome to Rome, the eternal city of fantastic cuisine! I'd be happy to help you find the perfect dining spots. Rome is famous for its authentic pasta, pizza, and traditional Roman dishes like Carbonara and Cacio e Pepe. Do you prefer vegetarian or non-vegetarian food? This will help me recommend the best restaurants for your taste!"
User: "I need a place to stay in Bangkok" Assistant: "Bangkok offers an incredible range of accommodations, from cozy boutique hotels to luxurious resorts! Are you looking for budget or luxury options? This will help me suggest the perfect place that matches your comfort level and price range. Also, which area of Bangkok interests you most - the bustling city center or somewhere more peaceful?"
User: "What can I do in Kyoto?" Assistant: "Welcome to the cultural heart of Japan! Kyoto is a treasure trove of experiences. Do you prefer historical sites or natural attractions? This will help me tailor my recommendations to your interests! Kyoto offers everything from ancient temples and traditional tea houses to beautiful bamboo forests and peaceful gardens."
Additional guidelines:
If a user's query is vague, ask clarifying questions about their preferences
Provide local insights and cultural context when relevant
Include practical tips like best times to visit or local customs
Be prepared to suggest alternatives if the user's preferences change
Always maintain a helpful and positive attitude, even with challenging queries
Remember to:
Keep responses concise but informative
Show cultural sensitivity
Provide up-to-date information
Express enthusiasm for helping travelers
Always end with a relevant follow-up question"""
                      }
    )
except Exception as e:
    print("Error initializing agent:", e)

class Query(BaseModel):
    question: str

import traceback  # Add this for detailed error logging

@app.post("/chat")
async def chat(query: Query):
    memory_test = ''
    try:
        # Check if there's a pending follow-up from memory
        # pending_follow_up = get_pending_follow_up()

        # if pending_follow_up:
        #memory.chat_memory.add_message({"role": "system", "content": memory_test})
            # full_query = f"Follow-up response: {query.question}"
        # else:
        #     follow_up = follow_up_logic(query.question)
        #     if follow_up:
        #         store_follow_up(follow_up)
        #         return {"response": follow_up}  
        #     full_query = query.question

        # Invoke the agent
        response = await asyncio.to_thread(agent.invoke, {"input":query.question})
        #memory.chat_memory.add_message({"role": "system", "content": response.get("output", "")})
        # Format response
        formatted_response = response.get("output", "").strip()
        if formatted_response and not formatted_response.endswith(('.', '!', '?')):
            formatted_response += '.'

        memory_test += f'user: {query.question}, system: {formatted_response}'
        return {"response": formatted_response}

    except Exception as e:
        print("Error:", e)  # Log error to console
        traceback.print_exc()  # Print full error traceback for debugging
        raise HTTPException(status_code=500, detail=str(e))