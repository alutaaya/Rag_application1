import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# --- 0. Load environment variables ---
load_dotenv()

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
    "TB_Lep":"1UUKe1PPgti_Gm6RgDq-kBexv2BgYXxTF","CKD.pdf":"1sOVGB7R1IEu3kWQrdd0IZCNmxXHJ3jWC",
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


# --- 2. Extract text and tables from PDF ---
def extract_text_and_tables(pdf_path):
    """
    Extract text and tables from a PDF file.
    Tables are extracted as text content (not numeric structure).
    Returns a list of LangChain Documents with metadata.
    """
    documents = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # --- Extract normal text ---
            text = page.extract_text()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "page": page_num,
                            "type": "text",
                        },
                    )
                )

            # --- Extract tables as text blocks ---
            tables = page.extract_tables()
            for table_idx, table in enumerate(tables):
                if not table:
                    continue
                # Convert table (list of lists) into readable text
                table_text = "\n".join([" | ".join(row) for row in table if any(row)])
                documents.append(
                    Document(
                        page_content=table_text,
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "page": page_num,
                            "type": f"table_{table_idx+1}",
                        },
                    )
                )

    return documents


# --- 3. Build or Load Vector Store ---
@st.cache_resource
def load_vectorstore():
    hf_embed = load_embeddings()

    # --- If vectorstore exists, load it ---
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            return FAISS.load_local(
                FAISS_INDEX_PATH, hf_embed, allow_dangerous_deserialization=True
            )
        except Exception:
            st.warning("Error loading FAISS index. Rebuilding index.")

    # --- Otherwise, extract from all PDFs in folder ---
    all_documents = []
    for pdf_file in os.listdir(PDF_FOLDER):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, pdf_file)
            all_documents.extend(extract_text_and_tables(pdf_path))

    # --- Split into manageable chunks ---
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    text_chunks = splitter.split_documents(all_documents)

    # --- Create vectorstore ---
    vectorstore = FAISS.from_documents(text_chunks, hf_embed)
    vectorstore.save_local(FAISS_INDEX_PATH)

    return vectorstore


@st.cache_resource
def load_llm():
    # ✅ safer access to secrets
    groq_api = st.secrets.get("groq_api") or os.getenv("groq_api")

    if not groq_api:
        st.error("❌ ERROR: groq_api not found. Please set it in Streamlit secrets or .env file.")
        return None

    return ChatGroq(
        model_name="openai/gpt-oss-120b",
        temperature=0,
        api_key=groq_api
    )

# --- 4. Core Functions ---
def retrieve_relevant_chunks(query, vectorstore):
    if vectorstore:
        return vectorstore.similarity_search(query, k=16)
    return []

def answer_query(query, llm, vectorstore):
    if not llm:
        return "Error: Language Model unavailable."
    if not vectorstore:
        return "Error: Vector store unavailable."

    docs = retrieve_relevant_chunks(query, vectorstore)
    if not docs:
        return "I could not find relevant information in the document."

    context = "\n\n".join([d.page_content for d in docs])
    prompt = (
        "You are a Ugandan chronic healthcare  assistant. Answer using ONLY the provided context. "
        "If the answer is not in the context, say you don't know. Be concise.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{query}\nAnswer:"
    )
    try:
        response = llm.invoke(prompt)
        return response.content.strip() if response else "Error: Empty response."
    except Exception as e:
        return f"Error invoking LLM: {e}"

# --- 5. Streamlit UI ---
def main():
    st.title("Uganda Chronic care Health Assistant Chatbot")
    st.write("Built by **Alfred Lutaaya** ")

    # ✅ debug: see which secrets are available
    st.write("DEBUG: Available secrets →", list(st.secrets.keys()))

    vectorstore = load_vectorstore()
    llm = load_llm()

    # --- Initialize chat history ---
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # --- User input ---
    user_input = st.text_input("Enter your question:")

    if st.button("Ask") and user_input:
        answer = answer_query(user_input, llm, vectorstore)
        st.session_state["chat_history"].append((user_input, answer))

    if st.button("Clear Chat"):
        st.session_state["chat_history"] = []

    # --- Display chat history ---
    for q, a in st.session_state["chat_history"]:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Assistant:** {a}")
        st.markdown("---")

if __name__ == "__main__":
    main()