# Local Semantic Document Search

A **fully local** semantic document search engine you can run on your laptop. No API keys, no paid services.

## Features

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector database**: Chroma (persistent on disk) in `./chroma_db/`
- **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Backend**: FastAPI + Uvicorn
- **Frontend**: single-file HTML/CSS/JS served by FastAPI
- **Upload button**: upload documents from the UI into `./documents/`
- **Offline-friendly**: supports running without network once models are available locally

## Supported file types

- PDF (`.pdf`)
- Word (`.docx`)
- Text (`.txt`)
- Markdown (`.md`)
- HTML (`.html`, `.htm`)

## Project structure

```text
semantic-search/
├── main.py           # FastAPI app + routes
├── indexer.py        # Parsing + chunking + embedding + Chroma storage
├── searcher.py       # ANN retrieval + cross-encoder reranking
├── static/
│   └── index.html    # UI
├── documents/        # Uploaded / user-added docs
├── chroma_db/        # Persistent Chroma DB
├── indexed_files.json
└── requirements.txt
```

## Quickstart

From the `semantic-search/` folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open `http://127.0.0.1:8000`.

## How indexing works

- The app reads documents from `./documents/`
- Text is chunked into ~400 token chunks with 50 token overlap (simple sliding window)
- Each chunk is embedded and stored in Chroma with metadata:
  - `source_file`
  - `chunk_index`
  - `chunk_text`
- Already-indexed files are tracked by filename in `indexed_files.json` (to avoid re-indexing on every run)

## Using the app

### Upload documents (recommended)

1. Click the **upload icon** (top-left button)
2. Select one or more files
3. The app uploads them into `./documents/` and triggers indexing

### Add documents manually

1. Copy files into `semantic-search/documents/`
2. Click **Re-index documents** (or call `POST /index`)

### Search

- Enter a query and press **Enter** or click **Search**
- Results show:
  - filename
  - relevance score (0–1)
  - matching chunk text

Keyboard shortcuts:

- **Enter**: search
- **Cmd/Ctrl+K**: focus the search box

## API

Base URL: `http://127.0.0.1:8000`

### `GET /`

Serves the UI.

### `POST /upload`

Uploads one or more files into `./documents/`.

- **Form field**: `files` (repeatable)
- **Allowed extensions**: `.pdf .docx .txt .md .html .htm`

### `POST /index`

Indexes any **new** documents in `./documents/`.

Response:

```json
{ "new_chunks_indexed": 123 }
```

### `POST /search`

Request:

```json
{ "query": "..." }
```

Response (top 5):

```json
[
  { "chunk_text": "...", "source_file": "file.pdf", "score": 0.842 }
]
```

### `GET /status`

Returns counts:

```json
{ "documents": 12, "chunks": 987 }
```

### `GET /files`

Lists files present in `./documents/` (not necessarily indexed yet).

## Offline mode (when Hugging Face is blocked)

Normally, the first run downloads two free models from Hugging Face:

- `sentence-transformers/all-MiniLM-L6-v2`
- `cross-encoder/ms-marco-MiniLM-L-6-v2`

If your network blocks `huggingface.co` (common behind strict proxies), do this once on a machine/network that *can* access Hugging Face:

```bash
python -m venv .venv
source .venv/bin/activate
pip install "huggingface_hub>=0.23,<1"

mkdir -p models
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
  repo_id="sentence-transformers/all-MiniLM-L6-v2",
  local_dir="models/all-MiniLM-L6-v2",
  local_dir_use_symlinks=False,
)
snapshot_download(
  repo_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
  local_dir="models/ms-marco-MiniLM-L-6-v2",
  local_dir_use_symlinks=False,
)
print("Done.")
PY
```

Then copy those folders into this repo:

```text
semantic-search/models/all-MiniLM-L6-v2/
semantic-search/models/ms-marco-MiniLM-L-6-v2/
```

Start the server in offline mode:

```bash
cd semantic-search
source .venv/bin/activate

export EMBED_MODEL_PATH="./models/all-MiniLM-L6-v2"
export RERANK_MODEL_PATH="./models/ms-marco-MiniLM-L-6-v2"
export LOCAL_FILES_ONLY=1

uvicorn main:app --reload --port 8000
```

## Troubleshooting

### “Address already in use”

Another server is already using that port. Start on a new port:

```bash
uvicorn main:app --reload --port 8003
```

### “uvicorn: command not found”

You didn’t activate the virtual environment:

```bash
cd semantic-search
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

### “Models not available (offline)”

Your network likely blocks `huggingface.co`. Follow the **Offline mode** section above.

### “No new documents to index”

Files are skipped by filename once indexed. If you edited a file but kept the same filename, rename it (or delete `indexed_files.json` to force a full re-index).

## Notes / limitations

- This is optimized for **local** usage and small-to-medium document sets.
- Chunking is intentionally simple (token sliding window) and can be improved later (headings, semantic boundaries, etc.).
- Scores are **not calibrated probabilities**; they’re normalized for UI friendliness.

## License

Choose a license for GitHub (e.g., MIT) if you plan to publish publicly.

