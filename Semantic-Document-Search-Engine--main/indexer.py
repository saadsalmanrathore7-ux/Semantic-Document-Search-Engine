import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Disable Chroma + HF telemetry noise (fully local app).
os.environ.setdefault("CHROMA_TELEMETRY", "FALSE")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import chromadb
from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_DIR = Path("./chroma_db")
DOCUMENTS_DIR = Path("./documents")
INDEXED_FILES_PATH = Path("./indexed_files.json")
COLLECTION_NAME = "documents"

MODEL_CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", "./models_cache"))
# Optional: point directly at a local folder containing the model.
# Example: export EMBED_MODEL_PATH=./models/all-MiniLM-L6-v2
EMBED_MODEL_PATH = os.environ.get("EMBED_MODEL_PATH", "").strip() or None
# If set to "1", loads models strictly from local disk (no network).
LOCAL_FILES_ONLY = os.environ.get("LOCAL_FILES_ONLY", "").strip() == "1"


_embedder: Optional[SentenceTransformer] = None
_tokenizer = None


def get_embedder() -> SentenceTransformer:
    """
    Load the embedding model once (CPU) and reuse it across calls.
    """
    global _embedder
    if _embedder is None:
        model_id = EMBED_MODEL_PATH or EMBED_MODEL_NAME
        logger.info("Loading embedding model: %s", model_id)
        try:
            MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            _embedder = SentenceTransformer(
                model_id,
                device="cpu",
                cache_folder=str(MODEL_CACHE_DIR),
            )
        except Exception as e:
            hint = (
                "Embedding model could not be loaded. If your network blocks Hugging Face, "
                "download the model once elsewhere and point to it locally via EMBED_MODEL_PATH, "
                "or set LOCAL_FILES_ONLY=1 and ensure it's present in the cache.\n"
                f"Tried model_id={model_id!r}, cache={str(MODEL_CACHE_DIR)!r}."
            )
            raise RuntimeError(hint) from e
    return _embedder


def get_tokenizer():
    """
    Use the same tokenizer family as the embedding model for approximate token counts.
    This keeps chunking closer to "tokens" than naive whitespace splitting.
    """
    global _tokenizer
    if _tokenizer is None:
        model_id = EMBED_MODEL_PATH or EMBED_MODEL_NAME
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=str(MODEL_CACHE_DIR),
            local_files_only=LOCAL_FILES_ONLY,
        )
    return _tokenizer


def get_collection():
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _load_indexed_files() -> Dict[str, bool]:
    if not INDEXED_FILES_PATH.exists():
        return {}
    try:
        with INDEXED_FILES_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): True for k in data.keys()}
        if isinstance(data, list):
            return {str(k): True for k in data}
        return {}
    except Exception:
        logger.exception("Failed to read indexed_files.json; starting fresh.")
        return {}


def _save_indexed_files(indexed: Dict[str, bool]) -> None:
    try:
        with INDEXED_FILES_PATH.open("w", encoding="utf-8") as f:
            json.dump(sorted(indexed.keys()), f, indent=2)
    except Exception:
        logger.exception("Failed to write indexed_files.json.")


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _parse_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: List[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def _parse_docx(path: Path) -> str:
    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(parts)


def _parse_html(path: Path) -> str:
    html = _safe_read_text(path)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def parse_document(path: Path) -> str:
    """
    Parse supported document types into plain text.
    If parsing fails, an exception is raised to be handled by the caller.
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _parse_pdf(path)
    if ext == ".docx":
        return _parse_docx(path)
    if ext in (".html", ".htm"):
        return _parse_html(path)
    if ext in (".txt", ".md"):
        return _safe_read_text(path)
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, chunk_tokens: int = 400, overlap_tokens: int = 50) -> List[str]:
    """
    Chunk text with a simple sliding window over tokenizer token IDs.
    No external chunking library required.
    """
    text = (text or "").strip()
    if not text:
        return []

    tokenizer = get_tokenizer()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    chunks: List[str] = []
    step = max(1, chunk_tokens - overlap_tokens)
    for start in range(0, len(token_ids), step):
        window = token_ids[start : start + chunk_tokens]
        if not window:
            continue
        chunk = tokenizer.decode(window, skip_special_tokens=True).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_tokens >= len(token_ids):
            break
    return chunks


def _iter_document_files(documents_dir: Path) -> Iterable[Path]:
    if not documents_dir.exists():
        return []
    exts = {".pdf", ".docx", ".txt", ".html", ".htm", ".md"}
    for p in sorted(documents_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def index_documents(documents_dir: Path = DOCUMENTS_DIR) -> int:
    """
    Scan ./documents for new files, chunk+embed them, and store in Chroma.
    Skips already-indexed files tracked in indexed_files.json (by filename).

    Returns the number of newly indexed chunks.
    """
    documents_dir = Path(documents_dir)
    documents_dir.mkdir(parents=True, exist_ok=True)

    indexed = _load_indexed_files()
    collection = get_collection()
    embedder = get_embedder()

    newly_indexed_chunks = 0

    for path in _iter_document_files(documents_dir):
        filename = path.name
        if indexed.get(filename):
            continue

        try:
            text = parse_document(path)
        except Exception:
            logger.exception("Failed to parse %s; skipping.", path)
            indexed[filename] = True  # prevent repeated crashes on every index run
            continue

        chunks = chunk_text(text, chunk_tokens=400, overlap_tokens=50)
        if not chunks:
            logger.warning("No text extracted from %s; skipping.", path)
            indexed[filename] = True
            continue

        try:
            embeddings = embedder.encode(
                chunks,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,  # cosine works best when normalized
            )

            ids = [f"{filename}::chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "source_file": filename,
                    "chunk_index": i,
                    "chunk_text": chunks[i],
                }
                for i in range(len(chunks))
            ]

            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
            )

            newly_indexed_chunks += len(chunks)
            indexed[filename] = True
            logger.info("Indexed %s (%d chunks).", filename, len(chunks))
        except Exception:
            logger.exception("Failed to embed/store %s; skipping.", path)
            indexed[filename] = True

    _save_indexed_files(indexed)
    return newly_indexed_chunks


def get_status() -> Dict[str, int]:
    """
    Return simple status numbers for the UI.
    """
    indexed = _load_indexed_files()
    collection = get_collection()
    try:
        chunk_count = int(collection.count())
    except Exception:
        chunk_count = 0
    return {"documents": len(indexed), "chunks": chunk_count}

