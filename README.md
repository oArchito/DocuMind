# 🧠 DocuMind — AI-Powered PDF RAG Studio

[![Python 3.12](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/Frontend-React%2019-61DAFB.svg)](https://react.dev/)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-ff69b4.svg)](https://github.com/facebookresearch/faiss)
[![Gemini 3.5](https://img.shields.io/badge/LLM-Gemini%203.5%20Flash-4285F4.svg)](https://ai.google.dev/)

> **DocuMind** is a production-grade Retrieval-Augmented Generation (RAG) web application that enables users to upload PDF documents and ask questions grounded strictly in their document's content with source passage citations.

---

## 🚀 Key Features

* **⚡ Ultra-Low Latency Vector Indexing**: Parallel multi-threaded embedding generation with FAISS similarity search.
* **🎯 Grounded Answer Generation**: Strictly constrained RAG prompts to eliminate hallucinations.
* **🎨 Studio-Grade Split Workspace Layout**: Dual-panel document control sidebar and interactive AI chat stream.
* **📊 Automatic Evaluation Suite**: Scorecard evaluation pipeline testing precision across multiple chunk size & top-k parameters.
* **🛡️ Resilient API Fallbacks**: Exponential backoff and multi-model fallback strategy.

---

## 🛠️ Architecture Overview

```
 ┌────────────────┐       ┌─────────────────┐       ┌──────────────────┐
 │  PDF Document  │ ────> │ PyPDF Extractor │ ────> │ Chunking Engine  │
 └────────────────┘       └─────────────────┘       └────────┬─────────┘
                                                             │
 ┌────────────────┐       ┌─────────────────┐                ▼
 │  User Question │ ────> │  FAISS Vector   │ <──── │ Gemini Embedding │
 └───────┬────────┘       │ Similarity Search│       └──────────────────┘
         │                └────────┬────────┘
         ▼                         ▼
 ┌──────────────────────────────────────────┐
 │  Context-Enriched Gemini 3.5 Flash RAG   │ ────> Structured Answer + Sources
 └──────────────────────────────────────────┘
```

---

## 💻 Tech Stack

* **Frontend**: React 19, Vite, Vanilla CSS Studio Tokens
* **Backend**: FastAPI, PyPDF, FAISS CPU, NumPy
* **AI & Vector Embeddings**: Google Gemini API (`gemini-3.5-flash`, `gemini-embedding-001`)

---

## 🚦 Quick Start

### 1. Backend Setup
```bash
cd backend
python -m venv venv
.\venv\Scripts\activate   # On Windows
pip install -r requirements.txt
```

Create a `.env` file in `backend/.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_TEXT_MODEL=gemini-3.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

Start the FastAPI server:
```bash
python -m uvicorn main:app --reload --port 8000
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173/](http://localhost:5173/) in your browser.

---

## 🔬 Running RAG Evaluation

Evaluate precision across test questions:
```bash
cd backend
python eval/run_eval.py
```

---

## 📜 License
MIT License © 2026 Archit Aggarwal
