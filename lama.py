# =========================
# Imports
# =========================
from llama_parse import LlamaParse
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate

import nest_asyncio
nest_asyncio.apply()

import os
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# =========================
# App & Environment Setup
# =========================
st.set_page_config(page_title="Uganda Chronic Care Assistant", layout="wide")

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY")

PDF_FOLDER = "pdfs"
FAISS_INDEX_PATH = "faiss_index"
os.makedirs(PDF_FOLDER, exist_ok=True)


# =========================
# Clinical Prompt Template
# =========================
CLINICAL_QA_PROMPT = PromptTemplate(
    """You are a clinical guideline assistant.

Answer the question using ONLY the retrieved document content.

Rules:
- Respond in at most 2 short sentences
- Give a direct clinical answer
- Do NOT add explanations, background, or extra recommendations
- If timing/frequency is asked, state only the timing and frequency

Question: {query_str}

Answer:"""
)


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


# =========================
# Utilities
# =========================
def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = next((v for k, v in response.cookies.items() if k.startswith("download_warning")), None)
    if token:
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            f.write(chunk)


# =========================
# Step 1: Ensure PDFs exist (cached)
# =========================
@st.cache_resource
def ensure_pdfs():
    for filename, file_id in PDF_FILES.items():
        path = os.path.join(PDF_FOLDER, filename)
        if not os.path.exists(path):
            download_file_from_google_drive(file_id, path)
    return True


# =========================
# Step 2: Parse PDFs → Nodes (cached)
# =========================
@st.cache_resource
def load_nodes():
    parser = LlamaParse(result_type="markdown")
    node_parser = SimpleNodeParser()

    all_documents = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.lower().endswith(".pdf"):
            docs = parser.load_data(os.path.join(PDF_FOLDER, filename))
            for d in docs:
                d.metadata["source_file"] = filename
            all_documents.extend(docs)

    return node_parser.get_nodes_from_documents(all_documents)


# =========================
# Step 3: Build FAISS Vector Store (cached)
# =========================
@st.cache_resource
def load_vectorstore(nodes):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    lc_documents = [
        Document(page_content=node.get_text(), metadata=node.metadata)
        for node in nodes
    ]

    return FAISS.from_documents(lc_documents, embeddings)


# =========================
# Step 4: Build Query Engine with Clinical Prompt (cached)
# =========================
@st.cache_resource
def load_query_engine(nodes):
    llm = OpenAI(api_key=OPENAI_API_KEY)

    index = VectorStoreIndex(
        nodes,
        embedding=OpenAIEmbedding(api_key=OPENAI_API_KEY)
    )

    return index.as_query_engine(
        llm=llm,
        text_qa_template=CLINICAL_QA_PROMPT
    )


# =========================
# App Execution
# =========================
ensure_pdfs()
nodes = load_nodes()
vectorstore = load_vectorstore(nodes)
query_engine = load_query_engine(nodes)

st.success("Clinical knowledge base loaded ✅")

query = st.text_input("Ask a clinical question:")

if query:
    with st.spinner("Retrieving guidance..."):
        response = query_engine.query(query)
        st.markdown("### Answer")
        st.write(response)
