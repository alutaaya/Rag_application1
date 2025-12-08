import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from unstructured.partition.pdf import partition_pdf
import warnings
from langchain_core.documents import Document          # 👈 Needed for wrapping chunks

warnings.filterwarnings('ignore')
import os
import sys

# Add Poppler bin folder to PATH
poppler_path = r"C:\Users\alutaaya\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin"
os.environ["PATH"] += os.pathsep + poppler_path


# --- 1. Setup folder and file info ---
PDF_FOLDER = "pdfs"
os.makedirs(PDF_FOLDER, exist_ok=True)

# Example: Multiple Google Drive files
PDF_FILES = {
    "Consolidated-HIV-and-AIDS-Guidelines-2022.pdf": "1rY_UE-sIw4f5Z5VUt0pyllPs7tSENsSr",
    "PrEP.pdf": "1n0Mtds2dSb6lCaJm6Ic8-NtaEIHnH5UQ",
    "NTLP_manual.pdf": "1SEPZ9j5zew9XcIeCdrXwzcopCulf_APZ",
    "UCG.pdf":"1f68UdsRdYwXW5DNN61pBNQXK7TkpMc0o",
    "PMDT.pdf":"1zhFrJC90olY7aaledw_RyKR_sC58XV2j",
    "HTS.pdf":"1mI8r0B2GmRGoWrJAEAOBZcpXb5znPmWs",
    "IPC.pdf":"1DKmCrueBly6jFtUP9Ox631jqzGDsR2tV",
    "TB_children.pdf":"1HUtgNMO_D-CK6ofLPf6egteHG7lhsd5S",
    "prevention.pdf":"1yTZ6JiB4ky8CcGK9tabkH3kLWCT2js4J",
    "TB_Lep":"1UUKe1PPgti_Gm6RgDq-kBexv2BgYXxTF",
    "CKD.pdf":"1sOVGB7R1IEu3kWQrdd0IZCNmxXHJ3jWC",
    "DSD.pdf":"1WRerkPmfRAzgPS234yP56aJ8zYXPwcjT"
}

# --- 2. Download PDFs from Google Drive ---
def download_from_gdrive(file_id, dest_path):
    """Downloads a file from Google Drive using its ID."""
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(response.content)
    print(f"✅ Downloaded {os.path.basename(dest_path)}")

# --- 3. Partition and chunk PDFs ---
def partition_and_chunk_pdf(file_path):
    """Partitions a PDF into manageable chunks."""
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy='hi_res',
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=2000,
        combine_text_under_n_chars=500,
        new_after_n_chars=6000,
    )
    return chunks

# --- 4. Process all PDFs ---
def process_all_pdfs(pdf_folder, pdf_files):
    """Downloads and partitions all PDFs into chunks."""
    all_chunks = {}

    for filename, file_id in pdf_files.items():
        pdf_path = os.path.join(pdf_folder, filename)

        # Download if missing
        if not os.path.exists(pdf_path):
            download_from_gdrive(file_id, pdf_path)

        print(f"🧩 Processing {filename}...")
        chunks = partition_and_chunk_pdf(pdf_path)
        all_chunks[filename] = chunks

    return all_chunks

# --- 5. Run processing ---
all_chunks = process_all_pdfs(PDF_FOLDER, PDF_FILES)

# Print summary
for fname, chunks in all_chunks.items():
    print(f"📘 {fname}: {len(chunks)} chunks")
    if chunks:
        # Show sample text from first chunk
        first_chunk_text = getattr(chunks[0], "text", str(chunks[0]))  # fallback if no .text
        print("Sample text:", first_chunk_text[:400], "..." if len(first_chunk_text) > 400 else "")