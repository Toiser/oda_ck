import os
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pypdf import PdfReader

PDF_DIR = os.path.join(os.path.dirname(__file__), "pdf_knowledge_base")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# Set up logging
logging.basicConfig(level=logging.INFO, format="[INGEST] %(asctime)s %(message)s")

def ingest_pdf(pdf_path, vectorstore, splitter):
    """Parse PDF, split, embed, and upsert into Chroma."""
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            logging.warning(f"No text found in {pdf_path}")
            return
        # Split into text chunks
        chunks = splitter.split_text(text)
        metadatas = [{"source": os.path.basename(pdf_path)} for _ in chunks]
        vectorstore.add_texts(texts=chunks, metadatas=metadatas)
        vectorstore.persist()
        logging.info(f"Ingested {pdf_path} ({len(chunks)} chunks)")
    except Exception as e:
        logging.error(f"Failed to ingest {pdf_path}: {e}")

class PDFHandler(FileSystemEventHandler):
    """Watches for new PDFs and ingests them."""
    def __init__(self, vectorstore, splitter):
        self.vectorstore = vectorstore
        self.splitter = splitter
    def on_created(self, event):
        try:
            if event.is_directory or not event.src_path.lower().endswith(".pdf"):
                return
            # Retry logic: wait until file is ready
            for attempt in range(5):
                try:
                    time.sleep(1 + attempt)
                    ingest_pdf(event.src_path, self.vectorstore, self.splitter)
                    break
                except Exception as e:
                    logging.error(f"Error ingesting {event.src_path} (attempt {attempt+1}): {e}")
                    if attempt == 4:
                        logging.error(f"Giving up on {event.src_path}")
        except Exception as e:
            logging.critical(f"UNHANDLED EXCEPTION in PDFHandler: {e}", exc_info=True)

def initial_ingest(vectorstore, splitter):
    """Ingest all PDFs in the folder at startup."""
    for fname in os.listdir(PDF_DIR):
        if fname.lower().endswith(".pdf"):
            try:
                ingest_pdf(os.path.join(PDF_DIR, fname), vectorstore, splitter)
            except Exception as e:
                logging.critical(f"UNHANDLED EXCEPTION during initial ingest of {fname}: {e}", exc_info=True)

def start_watcher(vectorstore, splitter):
    """Start the watchdog observer in a background thread."""
    initial_ingest(vectorstore, splitter)
    observer = Observer()
    observer.schedule(PDFHandler(vectorstore, splitter), PDF_DIR, recursive=False)
    observer.start()
    logging.info("Started PDF folder watcher.")
    return observer

if __name__ == "__main__":
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    start_watcher(vectorstore, splitter)
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        pass
