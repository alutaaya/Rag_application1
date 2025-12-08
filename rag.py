import os
import requests
import streamlit as st
from dotenv import load_dotenv

# LlamaParse + LlamaIndex
from llama_parse import LlamaParse
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import nest_asyncio
nest_asyncio.apply()


# -------------------------------------------------------------
# 0. Load environment variables
# -------------------------------------------------------------
load_dotenv()

# -------------------------------------------------------------
# 1. Setup folder and file info
# -------------------------------------------------------------
PDF_FOLDER = "pdfs"
os.makedirs(PDF_FOLDER, exist_ok=True)

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

# -------------------------------------------------------------
# 2. Download PDFs from Google Drive (if missing)
# -------------------------------------------------------------
def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None

    # Detect download confirmation token
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    # Write file to disk
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            f.write(chunk)

# ------------------
# Download loop
# ------------------
for filename, file_id in PDF_FILES.items():
    pdf_path = os.path.join(PDF_FOLDER, filename)
    if not os.path.exists(pdf_path):
        st.info(f"Downloading {filename}...")
        download_file_from_google_drive(file_id, pdf_path)
        st.success(f"{filename} downloaded successfully!")
    else:
        st.info(f"{filename} already exists.")

# -------------------------------------------------------------
# 3. Streamlit Page Config
# -------------------------------------------------------------
st.set_page_config(page_title="Uganda Chronic Care Assistant", layout="wide")

# -------------------------------------------------------------
# 4. Parse all PDFs using LlamaParse
# -------------------------------------------------------------
parser = LlamaParse(result_type="markdown")
node_parser = SimpleNodeParser()
all_documents = []

st.header("Parsing PDF documents...")

for filename in os.listdir(PDF_FOLDER):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(PDF_FOLDER, filename)
        st.info(f"Parsing: {filename}")

        # STEP 1: Parse PDF → returns Document objects
        docs = parser.load_data(pdf_path)

        # STEP 2: Add metadata to each parsed chunk
        for d in docs:
            d.metadata["source_file"] = filename

        # STEP 3: Combine all parsed outputs
        all_documents.extend(docs)

st.success("All PDFs parsed successfully!")

# -------------------------------------------------------------
# 5. Convert parsed docs → nodes
# -------------------------------------------------------------
nodes = node_parser.get_nodes_from_documents(all_documents)

# -------------------------------------------------------------
# 6. Create LanceDB Vector Store
# -------------------------------------------------------------
vector_store = LanceDBVectorStore(uri="lancedb_store")

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# -------------------------------------------------------------
# 7. Build the vector index
# -------------------------------------------------------------
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
    embed_model=embed_model,
)

# -------------------------------------------------------------
# 8. Create the Query Engine
# -------------------------------------------------------------
query_engine = index.as_query_engine(similarity_top_k=15)

# -------------------------------------------------------------
# -------------------------------------------------------------
# 9. Streamlit Chat UI
# -------------------------------------------------------------
st.header("📘 Uganda Chronic Care Assistant – Ask Questions")

# Initialize session state for storing chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# User input
user_query = st.text_input("Ask something from the HIV/Chronic Care Manuals")

# Search button
if st.button("Search"):
    if user_query:
        # Retrieve top-k relevant chunks from vector store
        docs = query_engine.retrieve(user_query)  # use query_engine's internal retrieval

        if docs:
            # Construct context
            context = "\n\n".join([d.page_content for d in docs])
            prompt = (
                "You are a Ugandan chronic healthcare assistant. Answer using ONLY the provided context. "
                "If the answer is not in the context, say you don't know. Be concise.\n\n"
                f"Context:\n{context}\n\nQuestion:\n{user_query}\nAnswer:"
            )

            # Query the engine using the constructed prompt
            response = query_engine.query(prompt)
        else:
            response = "I could not find relevant information in the document."

        # Save to session state
        st.session_state["chat_history"].append((user_query, response))

# Clear Outputs button
if st.button("Clear Outputs"):
    st.session_state["chat_history"] = []

# Display chat history
for q, r in st.session_state["chat_history"]:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Assistant:** {r}")
    st.markdown("---")