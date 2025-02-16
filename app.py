import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://localhost:8000/chat"

st.title("NamasteBot 🤖")
st.write("A multilingual chatbot powered by RAG.")

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages from history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Type your message...")
if user_input:
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Show loader while waiting for response
    with st.spinner("Thinking... 🤔"):
        response = requests.post(API_URL, json={"question": user_input})
        if response.status_code == 200:
            bot_response = response.json().get("response", "Error: No response received")
        else:
            bot_response = "Error: Unable to connect to backend"
    
    # Add bot response to chat history
    st.session_state["messages"].append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)
