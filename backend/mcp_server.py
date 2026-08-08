"""
mcp_server.py  –  MCP (Model Context Protocol) server for DocuMind.

This is a BONUS feature that wraps the RAG retrieval function as an MCP tool,
so AI assistants (like Claude, Copilot, etc.) can search your uploaded documents.

The MCP tool "search_document" takes a question and returns relevant passages.

Setup:
    pip install mcp
    python mcp_server.py

Or add to your MCP client config:
    {
      "mcpServers": {
        "documind": {
          "command": "python",
          "args": ["backend/mcp_server.py"]
        }
      }
    }
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add backend dir to path
sys.path.insert(0, os.path.dirname(__file__))

import faiss
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("documind.mcp")

# ── Data directory ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"

# ── Cached documents ───────────────────────────────────────────────────────
_documents = {}


def _load_documents():
    """Load all persisted documents from disk."""
    if not DATA_DIR.exists():
        return
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
                _documents[doc_dir.name] = {
                    "index": index,
                    "chunks": chunks,
                    "filename": meta.get("filename", "unknown.pdf"),
                }
        except Exception as e:
            logger.warning(f"Could not load {doc_dir.name}: {e}")


def _get_available_docs() -> list[dict]:
    """Return a summary of all available documents."""
    if not _documents:
        _load_documents()
    return [
        {"document_id": doc_id, "filename": doc["filename"], "num_chunks": len(doc["chunks"])}
        for doc_id, doc in _documents.items()
    ]


def _search(document_id: str, question: str, top_k: int = 4) -> list[dict]:
    """Search a document for relevant passages."""
    import rag  # lazy import to avoid circular issues

    if not _documents:
        _load_documents()

    if document_id not in _documents:
        raise ValueError(f"Document '{document_id}' not found. Available: {list(_documents.keys())}")

    doc = _documents[document_id]
    return rag.retrieve(question, doc["index"], doc["chunks"], k=top_k)


# ── MCP Server Setup ───────────────────────────────────────────────────────

def main():
    """Start the MCP server with the search_document tool."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print("ERROR: MCP SDK not installed. Run: pip install mcp")
        print("Then try again: python mcp_server.py")
        sys.exit(1)

    mcp = FastMCP(
        "DocuMind",
        description="Search uploaded PDF documents using RAG retrieval",
    )

    @mcp.tool()
    def list_documents() -> str:
        """List all available documents that have been uploaded and indexed."""
        docs = _get_available_docs()
        if not docs:
            return "No documents available. Upload a PDF via the DocuMind web app first."
        lines = ["Available documents:"]
        for d in docs:
            lines.append(f"  - {d['filename']} (ID: {d['document_id']}, {d['num_chunks']} chunks)")
        return "\n".join(lines)

    @mcp.tool()
    def search_document(document_id: str, question: str, top_k: int = 4) -> str:
        """
        Search an uploaded PDF document for passages relevant to a question.

        Args:
            document_id: The ID of the document (use list_documents to find IDs)
            question: The question or search query
            top_k: Number of passages to return (default 4)

        Returns:
            The most relevant passages from the document.
        """
        try:
            results = _search(document_id, question, top_k)
            if not results:
                return "No relevant passages found."

            lines = [f"Found {len(results)} relevant passage(s):\n"]
            for i, r in enumerate(results, 1):
                lines.append(f"--- Passage {i} (score: {r['score']}) ---")
                lines.append(r['chunk_text'])
                lines.append("")

            return "\n".join(lines)

        except ValueError as e:
            return str(e)
        except Exception as e:
            return f"Search failed: {e}"

    # Run the server
    mcp.run()


if __name__ == "__main__":
    main()
