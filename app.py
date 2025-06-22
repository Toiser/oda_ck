import os
import time
import logging
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from ingest import start_watcher, ingest_pdf
import faulthandler; faulthandler.enable()
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load environment variables
load_dotenv()
TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
PDF_DIR = os.path.join(os.path.dirname(__file__), "pdf_knowledge_base")

# Set up logging
logging.basicConfig(level=logging.INFO, format="[APP] %(asctime)s %(message)s")

# Embeddings and vector store (single instance)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Load local GPT-2 model and pipeline
GPT2_TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
GPT2_MODEL = AutoModelForCausalLM.from_pretrained("gpt2")
GPT2_PIPE = pipeline("text-generation", model=GPT2_MODEL, tokenizer=GPT2_TOKENIZER)

app = Flask(__name__)

def call_llm(prompt):
    result = GPT2_PIPE(prompt, max_new_tokens=160)
    answer = result[0]['generated_text']
    # Remove the prompt from the answer if it appears at the start
    if answer.startswith(prompt):
        answer = answer[len(prompt):].lstrip()
    return answer[:300]  # Truncate to 1600 characters

def generate_answer(question):
    """Retrieve top-3 chunks and call LLM with RAG prompt."""
    try:
        docs = vectorstore.similarity_search(question, k=3)
        if not docs:
            return "No knowledge base found. Please upload PDFs."
        context = "\n\n".join(f"<PDF>{d.page_content}</PDF>" for d in docs)
        prompt = (
            "You are a helpful crisis companion. Prefer information contained between the <PDF> tags.\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer: "
        )
        return call_llm(prompt)
    except Exception as e:
        logging.error(f"generate_answer error: {e}")
        return "Error retrieving answer. Please try again."

def download_and_ingest_pdf(url):
    """Download PDF from URL and ingest it."""
    try:
        fname = os.path.join(PDF_DIR, f"media_{int(time.time())}.pdf")
        r = requests.get(url, timeout=30)
        with open(fname, "wb") as f:
            f.write(r.content)
        ingest_pdf(fname, vectorstore, splitter)
        return True
    except Exception as e:
        logging.error(f"Failed to download/ingest PDF: {e}")
        return False

@app.route("/message", methods=["POST"])
def message():
    """Twilio webhook: handle incoming SMS messages."""
    from_number = request.form.get("From")
    body = request.form.get("Body", "").strip()
    num_media = int(request.form.get("NumMedia", 0))
    logging.info(f"Message from {from_number}: {body} (media: {num_media})")
    resp = MessagingResponse()
    if num_media > 0:
        media_url = request.form.get("MediaUrl0")
        if media_url and media_url.lower().endswith(".pdf"):
            if download_and_ingest_pdf(media_url):
                resp.message("PDF received and ingested! You can now ask questions about its content.")
            else:
                resp.message("Failed to ingest PDF. Please try again.")
        else:
            resp.message("Only PDF files are supported.")
    elif body:
        answer = generate_answer(body)
        resp.message(answer)
    else:
        resp.message("Send a question or attach a PDF.")
    return str(resp)

if __name__ == "__main__":
    try:
        start_watcher(vectorstore, splitter)
        app.run(host="0.0.0.0", port=5050)  # Changed port to 5050
    except Exception as e:
        import traceback
        logging.critical(f"FATAL ERROR in app.py: {e}\n{traceback.format_exc()}")
