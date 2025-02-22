# NamasteBot 

This project implements a multilingual AI chatbot using **LangChain**, **Google Generative AI (Gemini)**, **Google Search**, and a **Retrieval-Augmented Generation (RAG) pipeline**. The chatbot can answer user queries, retrieve knowledge from the RAG pipeline, and fetch real-time information via Google Search. It also intelligently asks follow-up questions when necessary.

---

## **Architecture Overview**

### **System Components:**
1. **User Query Processing**
   - The query is processed to detect follow-up questions and determine the best retrieval method.
2. **Follow-up Question Tracker**
   - Limits follow-ups to avoid excessive back-and-forth.
3. **RAG Pipeline**
   - Fetches multilingual responses from a pre-built knowledge base.
4. **Google Search API**
   - Retrieves real-time information when required.
5. **Google Gemini AI**
   - Generates conversational responses.
6. **FastAPI Server**
   - Exposes an API endpoint (`/chat`) for external integration.

---

## **Flow Diagram**

```plaintext
+--------------------+
|    User Input     |
+--------+---------+
         |
         v
+-------------------+
|  Follow-up Check |
+-------------------+
         |
         v
+----------------------------+
| Determine Search Strategy |
| (Google Search or RAG)    |
+----------------------------+
         |
         v
+----------------------------+
|  Fetch Relevant Response  |
| (Search or Knowledge Base)|
+----------------------------+
         |
         v
+-------------------+
|  Generate Reply  |
|  (Gemini AI)    |
+-------------------+
         |
         v
+-------------------+
|  Return Output   |
+-------------------+
```

---

## **Installation & Setup**

### **1. Clone the repository**
```sh
git clone https://github.com/your-repo/multilingual-chatbot.git
cd multilingual-chatbot
```

### **2. Install dependencies**
```sh
pip install -r requirements.txt
```

### **3. Set up environment variables**
Create a `.env` file and add your API keys:
```env
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_cse_id
```

### **4. Run the FastAPI server**
```sh
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## **API Usage**

### **Endpoint:**
```
POST /chat
```
### **Request Body:**
```json
{
  "question": "What are the best places to visit in Goa?"
}
```
### **Response:**
```json
{
  "response": "Do you prefer historical sites or natural attractions?"
}
```

---

## **Features**
✅ Multilingual query handling  
✅ Intelligent follow-up questions  
✅ Uses RAG for contextual knowledge retrieval  
✅ Google Search for real-time data  
✅ FastAPI backend for seamless API integration  

---

## **Future Enhancements**
- Improve response ranking based on query type
- Add caching for frequently asked queries
- Integrate user feedback for response refinement

---

### **Contributors**
Ankita Kalangutkar - [GitHub Profile](https://github.com/AnkitaSK)

Feel free to contribute and improve the project!

