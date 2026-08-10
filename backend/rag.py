"""
rag.py  –  The RAG (Retrieval-Augmented Generation) engine for DocuMind.

This file contains all the "brain" functions:
  1. extract_text   – pull text out of a PDF file
  2. chunk_text     – split long text into overlapping pieces
  3. embed          – turn text pieces into number-vectors using Gemini
  4. build_index    – create a FAISS search index from vectors
  5. retrieve       – find the most relevant chunks for a question
  6. generate_answer – ask Gemini to answer using only those chunks
"""

import os
import time
import logging
import numpy as np
import faiss
from pypdf import PdfReader
from google import genai
from dotenv import load_dotenv

# ── Load environment variables from .env ────────────────────────────────────
load_dotenv()

# ── Set up logging ──────────────────────────────────────────────────────────
logger = logging.getLogger("documind.rag")

# ── Read configuration ──────────────────────────────────────────────────────
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "")
TEXT_MODEL          = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.5-flash")
EMBEDDING_MODEL     = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM       = int(os.getenv("EMBEDDING_DIM", "768"))

# ── Create the Gemini client ────────────────────────────────────────────────
# We create it lazily so import doesn't fail when the key is missing.
_client = None

def _get_client():
    """Return a cached Gemini client, creating it on first call."""
    global _client
    if _client is None:
        if not GEMINI_API_KEY or GEMINI_API_KEY == "your-api-key-here":
            raise RuntimeError(
                "GEMINI_API_KEY is not set.  "
                "Open backend/.env and paste your key on the GEMINI_API_KEY line.  "
                "Get a free key at https://aistudio.google.com"
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# 1.  EXTRACT TEXT FROM PDF
# ─────────────────────────────────────────────────────────────────────────────

def extract_text(pdf_path: str) -> str:
    """
    Read a PDF file and return all its text as one big string.

    Think of it like copying all the text from every page of a book
    and pasting it into one long document.

    Args:
        pdf_path: path to the PDF file on disk

    Returns:
        The full text content of the PDF.

    Raises:
        ValueError: if the PDF contains no extractable text (e.g. scanned images).
    """
    reader = PdfReader(pdf_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    full_text = "\n".join(pages_text)
    if not full_text.strip():
        raise ValueError(
            "This PDF has no extractable text.  "
            "It may be a scanned document (images only).  "
            "DocuMind needs PDFs with actual text content."
        )
    return full_text


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CHUNK TEXT
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """
    Split a long text into smaller overlapping pieces ("chunks").

    Args:
        text:       the full text to split
        chunk_size: how many characters per chunk  (default 1000 for fast embedding)
        overlap:    how many characters to repeat between consecutive chunks (default 150)

    Returns:
        A list of text chunks.
    """
    if overlap >= chunk_size:
        overlap = chunk_size // 5

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 3.  EMBED TEXTS  (turn text into number-vectors using Gemini with parallel batching)
# ─────────────────────────────────────────────────────────────────────────────

def _embed_single_batch(batch: list[str], max_retries: int = 5) -> list:
    """Helper function to embed a single batch with retry logic."""
    client = _get_client()
    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
            )
            return [emb.values for emb in result.embeddings]
        except Exception as e:
            error_msg = str(e).lower()
            if ("429" in str(e) or "resource" in error_msg or "quota" in error_msg or "503" in error_msg) and attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(f"Rate limited (attempt {attempt+1}/{max_retries}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Embedding batch failed after retries")


def embed(texts: list[str], batch_size: int = 50, max_retries: int = 5) -> np.ndarray:
    """
    Convert a list of text strings into embedding vectors concurrently.
    Uses ThreadPoolExecutor to run embedding API calls in parallel.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype="float32")

    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    # If single batch (e.g. question embedding), run directly without pool overhead
    if len(batches) == 1:
        batch_res = _embed_single_batch(batches[0], max_retries=max_retries)
        return np.array(batch_res, dtype="float32")

    # Run batches concurrently across worker threads
    import concurrent.futures
    all_embeddings = [None] * len(batches)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(batches))) as executor:
        future_to_idx = {
            executor.submit(_embed_single_batch, batch, max_retries): idx
            for idx, batch in enumerate(batches)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            all_embeddings[idx] = future.result()

    # Flatten list of lists
    flattened = [vec for b in all_embeddings for vec in b]
    return np.array(flattened, dtype="float32")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  BUILD FAISS INDEX
# ─────────────────────────────────────────────────────────────────────────────

def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS search index from embedding vectors.

    FAISS (Facebook AI Similarity Search) lets us quickly find which chunks
    are most similar to a question.  Think of it like building a phonebook
    sorted by meaning instead of alphabetically.

    We use Inner Product (IP) search after normalising the vectors,
    which is equivalent to cosine similarity.

    Args:
        embeddings: 2-D numpy array from embed()

    Returns:
        A FAISS index ready for searching.
    """
    # Normalise vectors so inner product == cosine similarity
    faiss.normalize_L2(embeddings)

    # Create an index that does exact inner-product search
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


# ─────────────────────────────────────────────────────────────────────────────
# 5.  RETRIEVE RELEVANT CHUNKS
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(
    question: str,
    index: faiss.IndexFlatIP,
    chunks: list[str],
    k: int = 4,
) -> list[dict]:
    """
    Find the k chunks most relevant to the question.

    This is the "retrieval" step in RAG:  we turn the question into an
    embedding, then search the FAISS index for the closest chunk-embeddings.

    Args:
        question: the user's question (plain text)
        index:    the FAISS index built from the document's chunks
        chunks:   the original text chunks (same order as when indexed)
        k:        how many chunks to return  (default 4)

    Returns:
        A list of dicts, each with keys:
          - chunk_text: the text of the matching chunk
          - chunk_id:   the index of the chunk (0-based)
          - score:      cosine-similarity score (higher = more relevant)
    """
    # Embed the question (result is shape (1, dim))
    q_embedding = embed([question])
    faiss.normalize_L2(q_embedding)

    # Search the index
    scores, indices = index.search(q_embedding, min(k, len(chunks)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue  # FAISS returns -1 when there aren't enough results
        results.append({
            "chunk_text": chunks[idx],
            "chunk_id": int(idx),
            "score": round(float(score), 4),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6.  GENERATE ANSWER
# ─────────────────────────────────────────────────────────────────────────────

def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Ask Gemini to answer the question using ONLY the retrieved chunks.

    This is the "generation" step in RAG.  We build a prompt that includes
    the relevant text passages and tells the model:
      "Answer ONLY from this context.  If the answer isn't here, say so."

    This prevents the LLM from making things up ("hallucinating").

    Args:
        question:          the user's question
        retrieved_chunks:  list of dicts from retrieve()

    Returns:
        The model's answer as a plain string.
    """
    # Build the context block from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(f"[Passage {i}]\n{chunk['chunk_text']}")
    context = "\n\n".join(context_parts)

    # The system prompt constrains the model to answer only from context
    prompt = f"""You are DocuMind, an intelligent AI document assistant. Answer the user's question using ONLY the provided context.

RESPONSE FORMATTING & STYLE GUIDELINES:
- Format your response using clean, structured GitHub Markdown.
- Use clear section headers (e.g. `### Overview`, `### Key Features`, `### Tech Stack`) or bold topic titles.
- Place EVERY bullet point on its own new line using `- `.
- Highlight key terms, technologies, numbers, and dates using **bold text**.
- Use relevant emojis (e.g. 🛠️, 📌, 🚀, 📄) where helpful to make the output visual and clean.
- Include passage references naturally (e.g., *(Passage 1)*).
- If the answer is not in the context, state clearly: "I couldn't find information about this in the uploaded document."

CONTEXT:
{context}

USER QUESTION:
{question}

STRUCTURED ANSWER:"""

    client = _get_client()

    models_to_try = [TEXT_MODEL, "gemini-3.5-flash", "gemini-3.6-flash", "gemini-flash-latest"]
    # De-duplicate while preserving order
    seen = set()
    candidate_models = [m for m in models_to_try if not (m in seen or seen.add(m))]

    last_exception = None
    for model_name in candidate_models:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return response.text.strip()
            except Exception as e:
                last_exception = e
                err_str = str(e).lower()
                if "quota" in err_str or "resource_exhausted" in err_str:
                    logger.warning(f"Quota exceeded for {model_name}, trying fallback model...")
                    break  # Break inner retry loop to try next model in candidate_models
                elif ("503" in err_str or "429" in err_str or "unavailable" in err_str or "overloaded" in err_str) and attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    logger.warning(f"Gemini API busy for {model_name} (attempt {attempt+1}/{max_retries}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    break

    if last_exception:
        raise last_exception
