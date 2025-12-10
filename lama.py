from llama_parse import LlamaParse 
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex,StorageContext
import nest_asyncio
nest_asyncio.apply()
import os
import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.documents import Document


# --- 0. Load environment variables ---
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY")

# --- 1. Setup folder and file info ---
PDF_FOLDER = "pdfs"
os.makedirs(PDF_FOLDER, exist_ok=True)

# Example: Multiple Google Drive files
PDF_FILES = {
    "Consolidated-HIV-and-AIDS-Guidelines-2022.pdf": "1rY_UE-sIw4f5Z5VUt0pyllPs7tSENsSr",
    "PrEP.pdf": "1n0Mtds2dSb6lCaJm6Ic8-NtaEIHnH5UQ",
    "NTLP_manual.pdf": "1SEPZ9j5zew9XcIeCdrXwzcopCulf_APZ",
    "UCG.pdf":"1f68UdsRdYwXW5DNN61pBNQXK7TkpMc0o","PMDT.pdf":"1zhFrJC90olY7aaledw_RyKR_sC58XV2j",
    "HTS.pdf":"1mI8r0B2GmRGoWrJAEAOBZcpXb5znPmWs","IPC.pdf":"1DKmCrueBly6jFtUP9Ox631jqzGDsR2tV",
    "TB_children.pdf":"1HUtgNMO_D-CK6ofLPf6egteHG7lhsd5S","prevention.pdf":"1yTZ6JiB4ky8CcGK9tabkH3kLWCT2js4J",
   "CKD.pdf":"1sOVGB7R1IEu3kWQrdd0IZCNmxXHJ3jWC",
    "DSD.pdf":"1WRerkPmfRAzgPS234yP56aJ8zYXPwcjT"
}

def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive given a file ID."""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            f.write(chunk)

# --- 2. Download all PDFs if missing ---
for filename, file_id in PDF_FILES.items():
    path = os.path.join(PDF_FOLDER, filename)
    if not os.path.exists(path):
        st.info(f"Downloading {filename}…")
        download_file_from_google_drive(file_id, path)
        st.success(f"{filename} downloaded successfully!")
    else:
        st.info(f"{filename} already exists.")

# --- 2. Streamlit Page Config ---
st.set_page_config(page_title="Uganda Chronic care Assistant", layout="wide")

# --- Paths ---
FAISS_INDEX_PATH = "faiss_index"

# --- 1. Cache Embeddings ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

node_parser=SimpleNodeParser()
parser = LlamaParse(result_type="markdown")
all_documents = []    # list to store parsed outputs

node_parser = SimpleNodeParser()
parser = LlamaParse(result_type="markdown")
all_documents = []

# Loop through all PDFs in the folder
for filename in os.listdir(PDF_FOLDER):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(PDF_FOLDER, filename)
        st.info(f"Parsing {filename}...")

        # Parse PDF → returns a list of Document objects
        docs = parser.load_data(pdf_path)

        # Attach metadata to each chunk/page
        for d in docs:
            d.metadata["source_file"] = filename

        # Add parsed docs to global list
        all_documents.extend(docs)

st.success("All PDFs parsed successfully!")
nodes = node_parser.get_nodes_from_documents(all_documents)

# Convert LlamaIndex nodes to LangChain Document objects
lc_documents = [Document(page_content=node.get_text(), metadata=node.metadata) for node in nodes]



# Load embeddings
embeddings = load_embeddings()

# Create FAISS vector store from nodes
# Now pass these to FAISS
vectorstore = FAISS.from_documents(documents=lc_documents, embedding=embeddings)

# Optionally, save to disk for persistence
vectorstore.save_local("faiss_index")


query = st.text_input("Ask a question about the documents:")
if query:
    results = vectorstore.similarity_search(query, k=12)  # retrieve top 5 similar docs
    for r in results:
        st.write(r.page_content)


# Use OpenAI or Llama Cloud to answer queries
index = VectorStoreIndex(nodes, embedding=OpenAIEmbedding(api_key=OPENAI_API_KEY))
response = index.query(query)
st.write(response)


