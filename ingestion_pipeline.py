import os
import sys
import zipfile
import logging
import warnings
from pathlib import Path
from typing import List, Optional

# Suppress noisy symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure PyTorch is initialized before transformers
import torch
import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ----------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------
load_dotenv()

ARCHIVE_DIR = Path(".")                          # folder containing zip archives
EXTRACT_DIR = Path("data/raw")                   # directory where CSVs are unpacked
PERSIST_DIR = Path("db/faiss_index")             # FAISS index persistence directory
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
# Set MAX_ROWS_PER_CSV (e.g., 2000 per CSV for fast ingestion, or 0 for all rows)
MAX_ROWS_PER_CSV = int(os.getenv("MAX_ROWS_PER_CSV", "2000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "256"))

LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ----------------------------------------------------------------------
# 2. Pipeline Helpers
# ----------------------------------------------------------------------
def unzip_all(archive_dir: Path, out_dir: Path) -> List[Path]:
    """Extract all *.zip files in archive_dir to out_dir and return unique CSV paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_files = list(archive_dir.glob("*.zip"))
    
    if not zip_files:
        logging.warning(f"No zip files found in {archive_dir}. Checking for existing CSVs in {out_dir}...")
    else:
        for zip_path in zip_files:
            logging.info(f"Extracting '{zip_path.name}' into '{out_dir}'...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(out_dir)

    csv_files = sorted(list(set(out_dir.rglob("*.csv"))))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {out_dir} after extraction.")
    
    logging.info(f"Found {len(csv_files)} unique CSV file(s): {[f.name for f in csv_files]}")
    return csv_files


def _format_row(row: dict) -> str:
    """Format a dataframe row dictionary into a clean, searchable text string."""
    parts = []
    for col, val in row.items():
        if col in ("Unnamed: 0", "index") or pd.isna(val) or val == "":
            continue
        if isinstance(val, float):
            val_str = f"{val:.2f}" if not val.is_integer() else f"{int(val)}"
        else:
            val_str = str(val).strip()
        parts.append(f"{col}: {val_str}")
    return " | ".join(parts)


def _extract_metadata(row: dict, filename: str, row_idx: int) -> dict:
    """Extract standardized metadata fields for downstream filtering."""
    meta = {"source": filename, "row": row_idx}
    
    # Map common column aliases to standardized keys
    col_mapping = {
        "crop": ["crop", "crop_names", "Crop"],
        "state": ["state", "state_names", "State_Name", "Region"],
        "district": ["district", "district_names", "District_Name"],
        "season": ["season", "season_names", "Season"],
        "year": ["crop_year", "Crop_Year", "year"],
    }
    
    for key, aliases in col_mapping.items():
        for alias in aliases:
            if alias in row and not pd.isna(row[alias]) and str(row[alias]).strip() != "":
                meta[key] = str(row[alias]).strip()
                break
                
    return meta


def csv_to_documents(csv_paths: List[Path], max_rows_per_csv: Optional[int] = None) -> List[Document]:
    """Load CSVs, format rows into readable text, and create LangChain Documents."""
    docs = []
    
    for path in csv_paths:
        logging.info(f"Reading dataset '{path.name}'...")
        # Handle various encodings gracefully
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
            
        df = df.dropna(how="all")
        total_rows = len(df)
        
        if max_rows_per_csv and max_rows_per_csv > 0 and total_rows > max_rows_per_csv:
            logging.info(f"Sampling {max_rows_per_csv} of {total_rows:,} rows from '{path.name}' (set MAX_ROWS_PER_CSV=0 for full dataset)...")
            df = df.sample(n=max_rows_per_csv, random_state=42)
        else:
            logging.info(f"Processing all {total_rows:,} rows from '{path.name}'...")
            
        records = df.to_dict(orient="records")
        for i, row in enumerate(records):
            content = _format_row(row)
            if content:
                meta = _extract_metadata(row, path.name, i)
                docs.append(Document(page_content=content, metadata=meta))
                
    logging.info(f"Total documents prepared: {len(docs):,}")
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """Split documents using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " | ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logging.info(f"Generated {len(chunks):,} text chunk(s)")
    return chunks


def build_vectorstore(chunks: List[Document], persist_dir: Path) -> FAISS:
    """Embed chunks using HuggingFace sentence transformer and persist FAISS index."""
    logging.info(f"Initializing embedding model '{EMBED_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    persist_dir.parent.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Indexing {len(chunks):,} chunk(s) into FAISS in batches of {BATCH_SIZE}...")
    vectorstore = None
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
            
        if (i + len(batch)) % (BATCH_SIZE * 5) == 0 or (i + len(batch)) == len(chunks):
            logging.info(f"Indexed {i + len(batch):,} / {len(chunks):,} chunk(s)...")
            
    logging.info(f"Saving vectorstore index to '{persist_dir}'...")
    vectorstore.save_local(str(persist_dir))
    logging.info(f"Vector store successfully saved to '{persist_dir}'")
    return vectorstore


def verify_index(persist_dir: Path, query: str = "Rice crop yield in Kharif season") -> None:
    """Verify that the persisted index can be loaded and queried."""
    logging.info(f"Verifying index at '{persist_dir}' with test query: '{query}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    loaded_vs = FAISS.load_local(
        str(persist_dir),
        embeddings,
        allow_dangerous_deserialization=True
    )
    results = loaded_vs.similarity_search(query, k=2)
    logging.info(f"Verification query returned {len(results)} result(s):")
    for idx, doc in enumerate(results, 1):
        logging.info(f" [Result {idx}] (Source: {doc.metadata.get('source', 'unknown')}): {doc.page_content[:150]}...")


# ----------------------------------------------------------------------
# 3. Main Entrypoint
# ----------------------------------------------------------------------
def main() -> None:
    logging.info("Starting Agricultural Chatbot Ingestion Pipeline...")
    csv_files = unzip_all(ARCHIVE_DIR, EXTRACT_DIR)
    raw_docs = csv_to_documents(csv_files, max_rows_per_csv=MAX_ROWS_PER_CSV)
    chunks = split_documents(raw_docs)
    build_vectorstore(chunks, PERSIST_DIR)
    verify_index(PERSIST_DIR)
    logging.info("Ingestion pipeline completed successfully!")


# Alias for programmatic execution
run_ingestion = main


if __name__ == "__main__":
    main()