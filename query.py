import faiss
import pickle
import os
import requests
import json
import time
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. Configuration & Clients
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OR_API_KEY = os.getenv("OPENROUTER_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

# 2. Load Local Data (FAISS + Chunks with Metadata)
print("Loading vector database and embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("medical_db.index")

# Using the metadata-enriched pickle file
with open("chunks_with_meta.pkl", "rb") as f:
    data = pickle.load(f)
    chunks = data["chunks"]
    metas = data["metadata"]

def get_context(query, k=5):
    """Retrieves context with page numbers for citations."""
    query_vector = embed_model.encode([query])
    distances, indices = index.search(query_vector.reshape(1, -1).astype('float32'), k)
    
    context_parts = []
    for i in indices[0]:
        text = chunks[i]
        page = metas[i]['page']
        context_parts.append(f"[Source: Page {page}]\n{text}")
    
    return "\n\n".join(context_parts)

# --- PRIMARY: GROQ (Optimized for 2026) ---
def call_groq(context, question):
    # Llama 3.3 70B and R1 Distill are the best current options
    models = ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b"]
    
    for model in models:
        try:
            print(f"Checking Groq ({model})...")
            completion = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a clinical assistant. Use provided context to answer. Always cite the page number found in brackets."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ],
                temperature=0.1 # Low temperature for medical precision
            )
            return completion.choices[0].message.content, f"Groq ({model})"
        except Exception as e:
            if "429" in str(e):
                print(f"Groq {model} rate limited...")
            continue
    return None, None

# --- FALLBACK: OPENROUTER (Verified 2026 IDs) ---
def call_openrouter(context, question):
    # Free models with high context for document processing
    models = [
        "deepseek/deepseek-r1-0528:free", 
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemini-2.0-flash-exp:free"
    ]
    headers = {
        "Authorization": f"Bearer {OR_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Medical_RAG_Production"
    }
    
    for model in models:
        try:
            print(f"Checking OpenRouter ({model})...")
            res = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps({
                    "model": model,
                    "messages": [{"role": "user", "content": f"Use this text to answer: {context}\n\nQuestion: {question}"}]
                }), timeout=25
            )
            if res.status_code == 200:
                return res.json()['choices'][0]['message']['content'], f"OpenRouter ({model})"
        except:
            continue
    return None, None

def ask_medical_rag(question):
    context = get_context(question)
    
    # 1. Primary Attempt: Groq
    answer, provider = call_groq(context, question)
    
    # 2. Fallback Attempt: OpenRouter
    if not answer:
        print("Falling back to OpenRouter...")
        answer, provider = call_openrouter(context, question)
        
    if answer:
        return answer, provider
    return "Service temporarily unavailable. All free providers are at capacity.", "None"

# 3. Interactive Loop
print("\n" + "="*50)
print("PRODUCTION MEDICAL RAG READY (GROQ + OPENROUTER)")
print("="*50)

while True:
    user_query = input("\n[User]: ")
    if user_query.lower() in ['quit', 'exit', 'q']: break
    
    print("\n[AI]: Analyzing book and generating response...")
    ans, prov = ask_medical_rag(user_query)
    
    print(f"\n--- RESPONSE (Source: {prov}) ---")
    print(ans)
    print("-" * 50)