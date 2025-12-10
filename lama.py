# =========================
# Imports
# =========================
import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv

from llama_parse import LlamaParse
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate

import nest_asyncio
nest_asyncio.apply()

# =========================
# App & Environment Setup
# =========================
st.set_page_config(page_title="Uganda Chronic Care Assistant", layout="wide")
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY")  # <- included
PDF_FOLDER = "pdfs"
INDEX_PATH = "rag_index"
os.makedirs(PDF_FOLDER, exist_ok=True)

# =========================
# PDF Source Registry
# =========================
PDF_FILES = {
    "Consolidated-HIV-and-AIDS-Guidelines-2022.pdf": "1rY_UE-sIw4f5Z5VUt0pyllPs7tSENsSr",
    "PrEP.pdf": "1n0Mtds2dSb6lCaJm6Ic8-NtaEIHnH5UQ",
    "NTLP_manual.pdf": "1SEPZ9j5zew9XcIeCdrXwzcopCulf_APZ",
    "UCG.pdf": "1f68UdsRdYwXW5DNN61pBNQXK7TkpMc0o",
    "PMDT.pdf": "1zhFrJC90olY7aaledw_RyKR_sC58XV2j",
    "HTS.pdf": "1mI8r0B2GmRGoWrJAEAOBZcpXb5znPmWs",
    "IPC.pdf": "1DKmCrueBly6jFtUP9Ox631jqzGDsR2tV",
    "TB_children.pdf": "1HUtgNMO_D-CK6ofLPf6egteHG7lhsd5S",
    "prevention.pdf": "1yTZ6JiB4ky8CcGK9tabkH3kLWCT2js4J",
    "CKD.pdf": "1sOVGB7R1IEu3kWQrdd0IZCNmxXHJ3jWC",
    "DSD.pdf": "1WRerkPmfRAzgPS234yP56aJ8zYXPwcjT"
}

def download_pdf(file_id, destination):
    """Download a PDF from Google Drive given file ID."""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            f.write(chunk)

# =========================
# Download PDFs
# =========================
st.info("Checking PDF files...")
for filename, file_id in PDF_FILES.items():
    path = os.path.join(PDF_FOLDER, filename)
    if not os.path.exists(path):
        st.info(f"Downloading {filename}...")
        download_pdf(file_id, path)
        st.success(f"{filename} downloaded successfully!")
    else:
        st.info(f"{filename} already exists.")

# =========================
# Parse PDFs
# =========================
parser = LlamaParse(result_type="markdown", api_key=LLAMA_CLOUD_API_KEY)  # Use Llama API key
node_parser = SimpleNodeParser()
all_docs = []

for filename in os.listdir(PDF_FOLDER):
    if filename.lower().endswith(".pdf"):
        path = os.path.join(PDF_FOLDER, filename)
        st.info(f"Parsing {filename}...")
        try:
            docs = parser.load_data(path)
            for d in docs:
                d.metadata["source_file"] = filename
            all_docs.extend(docs)
            st.success(f"{filename} parsed successfully!")
            time.sleep(1)
        except Exception as e:
            st.error(f"Failed to parse {filename}: {e}")

nodes = node_parser.get_nodes_from_documents(all_docs)

# =========================
# Build or load index
# =========================
if os.path.exists(INDEX_PATH):
    index = VectorStoreIndex.load_from_disk(INDEX_PATH)
else:
    index = VectorStoreIndex(nodes, embedding=OpenAIEmbedding(api_key=OPENAI_API_KEY))
    index.save_to_disk(INDEX_PATH)

# =========================
# Setup LLM & Prompt
# =========================
llm = OpenAI(api_key=OPENAI_API_KEY)

CLINICAL_PROMPT = PromptTemplate(
    """You are a clinical guideline assistant.
Answer in at most 2 short sentences, using ONLY the documents.
Question: {query_str}
Answer:"""
)

query_engine = index.as_query_engine(
    llm=llm,
    text_qa_template=CLINICAL_PROMPT
)

# =========================
# Streamlit UI
# =========================
st.title("Uganda Chronic Care Assistant (RAG)")
query = st.text_input("Ask a question about the clinical guidelines:")

if query:
    with st.spinner("Fetching answer..."):
        try:
            response = query_engine.query(query)
            st.markdown(f"**Answer:** {response.response}")
        except Exception as e:
            st.error(f"Failed to get answer: {e}")
