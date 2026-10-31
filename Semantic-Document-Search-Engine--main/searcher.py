import logging
import math
import os
from typing import Dict, List

from sentence_transformers import CrossEncoder, SentenceTransformer

from indexer import get_collection, get_embedder


logger = logging.getLogger(__name__)

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker: CrossEncoder | None = None

# Optional: point directly at a local folder containing the reranker model.
# Example: export RERANK_MODEL_PATH=./models/ms-marco-MiniLM-L-6-v2
RERANK_MODEL_PATH = os.environ.get("RERANK_MODEL_PATH", "").strip() or None
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "./models_cache")


def get_reranker() -> CrossEncoder:
    """
    Load the cross-encoder once (CPU) and reuse it across calls.
    """
    global _reranker
    if _reranker is None:
        model_id = RERANK_MODEL_PATH or RERANK_MODEL_NAME
        logger.info("Loading reranker model: %s", model_id)
        try:
            _reranker = CrossEncoder(model_id, device="cpu", cache_dir=MODEL_CACHE_DIR)
        except Exception as e:
            hint = (
                "Reranker model could not be loaded. If your network blocks Hugging Face, "
                "download the model once elsewhere and point to it locally via RERANK_MODEL_PATH.\n"
                f"Tried model_id={model_id!r}, cache={MODEL_CACHE_DIR!r}."
            )
            raise RuntimeError(hint) from e
    return _reranker


def search(query: str, top_k: int = 5) -> List[Dict]:
    """
    1) Embed query
    2) Retrieve top 20 from Chroma (cosine)
    3) Rerank with cross-encoder
    4) Return top 5: chunk_text, source_file, score (0-1)
    """
    query = (query or "").strip()
    if not query:
        return []

    collection = get_collection()
    embedder: SentenceTransformer = get_embedder()
    reranker = get_reranker()

    try:
        q_emb = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    except Exception:
        logger.exception("Failed to embed query.")
        return []

    try:
        res = collection.query(
            query_embeddings=[q_emb],
            n_results=20,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        logger.exception("Chroma query failed.")
        return []

    documents = (res.get("documents") or [[]])[0]
    metadatas = (res.get("metadatas") or [[]])[0]
    if not documents or not metadatas:
        return []

    pairs = [(query, doc) for doc in documents]
    try:
        scores = reranker.predict(pairs).tolist()  # higher is better
    except Exception:
        logger.exception("Reranking failed.")
        return []

    scored = []
    for doc, meta, s in zip(documents, metadatas, scores):
        # Convert cross-encoder logits-ish scores to a 0..1-ish number for UI.
        # This is not a calibrated probability, but it's stable and user-friendly.
        try:
            s_float = float(s)
            if not math.isfinite(s_float):
                s_float = -50.0
        except Exception:
            s_float = -50.0

        score_01 = float(1.0 / (1.0 + math.exp(-s_float)))
        if not math.isfinite(score_01):
            score_01 = 0.0
        scored.append(
            {
                "chunk_text": meta.get("chunk_text") or doc,
                "source_file": meta.get("source_file") or "unknown",
                "score": score_01,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

