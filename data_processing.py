import os
import pickle
import sys
from config import load_configuration
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load configuration
openai_api_key, root_dir = load_configuration()

def ingest_data(root_dir):
    # Load documents
    docs = []
    print(f"Checking {root_dir} for documents...")
    for dirpath, dirnames, filenames in os.walk(root_dir):
        print(f"Current directory path: {dirpath}")
        print(f"Subdirectories: {dirnames}")
        print(f"Files: {filenames}\n")
        for file in filenames:
            if file.endswith('.py') and '/.venv/' not in dirpath:
                try:
                    loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                    loaded_docs = loader.load()
                    print(f"Loaded {len(loaded_docs)} documents from {file}")
                    docs.extend(loaded_docs)
                except Exception as e: 
                    print(f"Error loading documents from {file}: {e}")
    if len(docs) == 0:
        print(f"No documents found in {root_dir}. Exiting...")
        sys.exit()
    else:
        print(f"Loaded total {len(docs)} documents from {root_dir}")


    # Split repository documents into text chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    print(f"Generated {len(texts)} text chunks")

    # Init embeddings object
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Send text chunks to OpenAI Embeddings API
    db = FAISS.from_documents(texts, embeddings)

    # Save vectorstore
    with open("db.pkl", "wb") as f:
        pickle.dump(db, f)
