import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
import configparser

#------------------------------------------------------------------------------

config = configparser.ConfigParser()
config.read("config.ini")


#  Read a PDF ----------------------------------------------------------------
def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text

#------------------------------------------------------------------------------
# Embeddings model
embeddings = OllamaEmbeddings(model=config["MODELS"]["embedder_model"])

# 2  Chunk text 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)


db = Chroma(
    collection_name=config["DATABASE"]["collection_name"],
    embedding_function=embeddings,
    persist_directory= config["DATABASE"]["persist_directory"]
)
 
# read the pdf, get chunks, store in the vector store.
 
pdf_folder = config["DATABASE"]["pdf_folder"]

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, file)
        print(f"Processing: {pdf_path}")

        # Read text
        text = read_pdf(pdf_path)

        # Chunk text
        chunks = text_splitter.split_text(text)

        # Add metadata (optional but useful)
        metadatas = [{"source": file} for _ in chunks]

        # Add the embeddings vectors to Chroma.
        db.add_texts(chunks, metadatas=metadatas)
        print(f"File {file} is chunked and stored in the vectorstore")

print("All PDFs have been ingested to Chroma.")
