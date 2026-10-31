import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from indexer import DOCUMENTS_DIR, get_status, index_documents
from searcher import get_reranker, search


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("semantic-search")


app = FastAPI(title="Local Semantic Document Search")


STATIC_INDEX = Path(__file__).parent / "static" / "index.html"


class SearchRequest(BaseModel):
    query: str


@app.on_event("startup")
def _startup() -> None:
    """
    Keep startup fast: do not download models or index documents here.
    Models and indexing will happen on first use (search/index endpoints).
    """
    logger.info("Startup complete (model warm-up and indexing deferred).")


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    if not STATIC_INDEX.exists():
        raise HTTPException(status_code=500, detail="static/index.html not found")
    return HTMLResponse(STATIC_INDEX.read_text(encoding="utf-8"))


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """
    Upload one or more files into ./documents for indexing.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    saved: List[Dict[str, Any]] = []
    for f in files:
        # Prevent path traversal and weird absolute paths.
        safe_name = Path(f.filename or "").name
        if not safe_name:
            saved.append({"filename": f.filename, "saved": False, "error": "Missing filename"})
            continue

        dest = DOCUMENTS_DIR / safe_name
        try:
            data = await f.read()
            dest.write_bytes(data)
            saved.append({"filename": safe_name, "saved": True, "bytes": len(data)})
        except Exception as e:
            logger.exception("Failed to save upload: %s", safe_name)
            saved.append({"filename": safe_name, "saved": False, "error": str(e)})

    return {"saved": saved}


@app.get("/files")
def list_files() -> Dict[str, Any]:
    """
    List files present in ./documents (not necessarily indexed).
    """
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for p in sorted(DOCUMENTS_DIR.glob("*")):
        if p.is_file():
            try:
                size = p.stat().st_size
            except Exception:
                size = None
            items.append({"name": p.name, "bytes": size})
    return {"files": items}


@app.post("/index")
def reindex() -> Dict[str, Any]:
    try:
        new_chunks = index_documents()
        return {"new_chunks_indexed": new_chunks}
    except Exception as e:
        logger.exception("Re-index failed.")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def do_search(req: SearchRequest):
    try:
        results = search(req.query, top_k=5)
        return results
    except Exception as e:
        logger.exception("Search failed.")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
def status() -> Dict[str, int]:
    try:
        return get_status()
    except Exception as e:
        logger.exception("Status failed.")
        raise HTTPException(status_code=500, detail=str(e))

