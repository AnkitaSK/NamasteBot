# NamasteBot - Multilingual AI Assistant

## Overview
NamasteBot is a powerful multilingual AI assistant that integrates **real-time search, knowledge retrieval, and advanced language generation** to provide accurate and context-aware responses. It leverages:

- **Google Search** for live updates
- **RAG pipeline** for structured knowledge retrieval
- **Gemini AI** for natural language responses
- **Conversation memory** for maintaining context
- **Follow-up question detection** for engaging interactions

This combination of tools ensures **relevant, personalized, and multilingual assistance** for various queries, including travel, dining, and general information.

---

## Architecture Overview

NamasteBot is built using **FastAPI**, **LangChain**, **Google Generative AI (Gemini-1.5-Pro)**, and a **Retrieval-Augmented Generation (RAG) pipeline** to deliver intelligent responses.

### **1. Tools and Their Roles**

| Tool Name                | Purpose |
|--------------------------|---------|
| **Google Search**        | Fetches **real-time** information from the web. |
| **RAG**                  | Retrieves **domain-specific** knowledge. |
| **Google Generative AI** | Generates **natural, structured** responses. |
| **Conversation Memory**  | Stores **chat history** for context-awareness. |
| **Follow-up Detector**   | Asks **clarifying questions** before responding. |

---

### **2. Tools in Detail**

#### **Google Search (Real-time Information Retrieval)**  
- Uses `GoogleSearchResults` from `langchain_google_community` to fetch fresh data from the web.
- Example Use Case: "current weather in Berlin."

#### **RAG Pipeline (Knowledge Base Retrieval)**  
- Custom module (`rag_pipeline`) integrated with **ChromaDB** and **MiniLM embeddings**.
- Example Use Case: "Places to visit in Goa"

#### **Google Generative AI (LLM for Response Generation)**  
- Uses **Gemini-1.5-Pro** to generate structured, multilingual responses.
- Processes chat history and user queries.

#### **Conversation Memory (Context Retention)**  
- Uses `ConversationBufferMemory` from `langchain.memory` to store past messages.
- Helps maintain logical flow in conversations.

#### **Follow-up Question Detector (Regex-based Classification)**  
- Identifies whether a follow-up question is needed before responding.
- Example: If the user asks, "Recommend a restaurant," it first asks:  
  **"Do you prefer vegetarian or non-vegetarian food?"**

---

## **Conclusion**
NamasteBot combines **real-time search, knowledge retrieval, and advanced AI-powered conversations** to deliver **accurate, engaging, and multilingual** responses. By leveraging **Google Search, RAG, and Gemini AI**, it ensures a seamless and intelligent user experience. 🚀

