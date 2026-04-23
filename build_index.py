from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def process_pdf():
    # 1. Load PDF
    loader = PyPDFLoader("dmv.pdf")
    documents = loader.load()

    # 2. Split into chunks (500 chars, 50 overlap)
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(docs)}")

    # 3. Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 4. Create FAISS vector store
    db = FAISS.from_documents(docs, embeddings)

    # 5. Save index locally
    db.save_local("faiss_index")

    print("✅ FAISS index created successfully!")

if __name__ == "__main__":
    process_pdf()
