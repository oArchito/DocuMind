"""
main.py  –  The FastAPI web server for DocuMind.

This file defines three API endpoints:
  POST /upload   – upload a PDF and index it for search
  POST /ask      – ask a question about an uploaded PDF
  GET  /health   – simple health check

Run it with:
    uvicorn main:app --reload --port 8000
"""

import os
import uuid
import json
import logging
import shutil
from pathlib import Path

import faiss
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import rag  # our RAG engine (rag.py in the same folder)

# ── Load .env ───────────────────────────────────────────────────────────────
load_dotenv()

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("documind.api")

# ── Data directory for persisting FAISS indexes ─────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── In-memory store: document_id → {index, chunks, filename} ───────────────
documents: dict = {}


def _load_persisted():
    """On startup, reload any previously persisted documents from disk."""
    for doc_dir in DATA_DIR.iterdir():
        if not doc_dir.is_dir():
            continue
        try:
            meta_path = doc_dir / "meta.json"
            index_path = doc_dir / "index.faiss"
            chunks_path = doc_dir / "chunks.json"
            if meta_path.exists() and index_path.exists() and chunks_path.exists():
                meta = json.loads(meta_path.read_text())
                index = faiss.read_index(str(index_path))
                chunks = json.loads(chunks_path.read_text())
                documents[doc_dir.name] = {
                    "index": index,
                    "chunks": chunks,
                    "filename": meta.get("filename", "unknown.pdf"),
                }
                logger.info(f"Loaded persisted document: {doc_dir.name}")
        except Exception as e:
            logger.warning(f"Could not load {doc_dir.name}: {e}")

_load_persisted()


# ── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="DocuMind API",
    description="Upload a PDF, then ask questions answered from its content.",
    version="1.0.0",
)

# ── CORS (so the React frontend can call us) ────────────────────────────────
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ───────────────────────────────────────────────

class AskRequest(BaseModel):
    """The JSON body for the /ask endpoint."""
    document_id: str
    question: str
    top_k: int = 4  # optional, defaults to 4


class SourceChunk(BaseModel):
    chunk_text: str
    chunk_id: int
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class UploadResponse(BaseModel):
    document_id: str
    num_chunks: int
    filename: str


class HealthResponse(BaseModel):
    status: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Simple health check – useful for deployment monitors."""
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    """
    Upload a PDF, extract its text, chunk and embed it, and build a
    searchable FAISS index.  Returns a document_id to use in /ask.
    """

    # ── Validate file type ──────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted.  Please upload a .pdf file.",
        )

    # ── Validate file size (max 10 MB) ──────────────────────────────────
    contents = await file.read()
    max_size = 10 * 1024 * 1024  # 10 MB
    if len(contents) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File is too large ({len(contents)/(1024*1024):.1f} MB).  "
                   "Maximum allowed size is 10 MB.",
        )

    # ── Save to a temporary path ────────────────────────────────────────
    doc_id = str(uuid.uuid4())
    doc_dir = DATA_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    temp_pdf = doc_dir / "upload.pdf"
    temp_pdf.write_bytes(contents)

    try:
        # 1. Extract text
        logger.info(f"Extracting text from: {file.filename}")
        text = rag.extract_text(str(temp_pdf))

        # 2. Chunk
        chunks = rag.chunk_text(text)
        logger.info(f"Created {len(chunks)} chunks")

        # 3. Embed (calls the Gemini API)
        logger.info("Embedding chunks via Gemini API...")
        embeddings = rag.embed(chunks)

        # 4. Build FAISS index
        index = rag.build_index(embeddings)
        logger.info("FAISS index built")

        # 5. Store in memory
        documents[doc_id] = {
            "index": index,
            "chunks": chunks,
            "filename": file.filename,
        }

        # 6. Persist to disk so we survive restarts
        faiss.write_index(index, str(doc_dir / "index.faiss"))
        (doc_dir / "chunks.json").write_text(json.dumps(chunks))
        (doc_dir / "meta.json").write_text(json.dumps({
            "filename": file.filename,
            "num_chunks": len(chunks),
        }))

        return UploadResponse(
            document_id=doc_id,
            num_chunks=len(chunks),
            filename=file.filename,
        )

    except ValueError as e:
        # Clean up on failure
        shutil.rmtree(doc_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(e))

    except RuntimeError as e:
        shutil.rmtree(doc_dir, ignore_errors=True)
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        shutil.rmtree(doc_dir, ignore_errors=True)
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Ask a question about an uploaded document.
    Retrieves the most relevant chunks and generates an answer.
    """

    # ── Validate document_id ────────────────────────────────────────────
    if req.document_id not in documents:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{req.document_id}' not found.  "
                   "Please upload the PDF first using /upload.",
        )

    doc = documents[req.document_id]

    try:
        # Retrieve relevant chunks
        sources = rag.retrieve(
            question=req.question,
            index=doc["index"],
            chunks=doc["chunks"],
            k=req.top_k,
        )

        # Generate answer from chunks
        answer = rag.generate_answer(req.question, sources)

        return AskResponse(
            answer=answer,
            sources=[SourceChunk(**s) for s in sources],
        )

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        logger.exception("Ask failed")
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")
