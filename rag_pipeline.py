# rag_pipeline.py

import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma  # Updated import path
from langchain.chains import RetrievalQA
from langdetect import detect
from deep_translator import GoogleTranslator

# Configure Google Gemini API
GOOGLE_API_KEY = 'AIzaSyBNu0Ea8aWQ1JnvXDOqSsnmH8Q2xZS7qww' #os.getenv("GOOGLE_API_KEY")
llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=GOOGLE_API_KEY)

# Load and Process PDF
loader = PyPDFLoader("Haridwar-Travel-Guide-by-ixigo.com.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Create Vector Store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = Chroma.from_documents(docs, embedding_model, persist_directory="./tmp/chroma_db")
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Create RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Multilingual Query Handling
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def translate_to_english(text, lang):
    if lang != "en":
        return GoogleTranslator(source=lang, target="en").translate(text)
    return text

def translate_back(text, lang):
    if lang != "en":
        return GoogleTranslator(source="en", target=lang).translate(text)
    return text

def rag_pipeline(query):
    # detected_lang = detect_language(query)
    # print(f"Detected Language: {detected_lang}")

    # translated_query = translate_to_english(query, detected_lang)
    # print(f"Translated Query: {translated_query}")

    response = qa_chain.invoke(query)
    result_text = response["result"]
    print(f"Raw Response: {result_text}")

    # final_response = translate_back(result_text, detected_lang)
    # print(f"Final Translated Response: {final_response}")

    return result_text
