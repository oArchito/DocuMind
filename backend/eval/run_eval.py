"""
run_eval.py  –  Evaluate the DocuMind RAG system.

This script:
  1. Generates a sample PDF (if it doesn't exist)
  2. Uploads it through the RAG pipeline (same functions the API uses)
  3. Asks 15 questions
  4. Checks if each answer contains the expected keyword
  5. Prints a scorecard

Usage:
    python run_eval.py                           # defaults: chunk_size=500, top_k=4
    python run_eval.py --chunk_size 300 --top_k 2
    python run_eval.py --run_all                 # run 9 configurations & save RESULTS.md
"""

import os
import sys
import json
import argparse
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add the backend directory to the path so we can import rag.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import rag  # noqa: E402


def run_evaluation(chunk_size: int = 500, overlap: int = 100, top_k: int = 4, verbose: bool = True):
    """
    Run the full evaluation pipeline.

    Returns:
        tuple: (correct_count, total_count, results_list)
    """
    eval_dir = os.path.dirname(__file__)
    pdf_path = os.path.join(eval_dir, "sample_handbook.pdf")
    questions_path = os.path.join(eval_dir, "questions.json")

    # ── Generate sample PDF if needed ──
    if not os.path.exists(pdf_path):
        if verbose:
            print("📄 Generating sample PDF...")
        from generate_sample_pdf import create_pdf
        create_pdf(pdf_path)

    # ── Load questions ──
    with open(questions_path, "r") as f:
        questions = json.load(f)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  DocuMind RAG Evaluation")
        print(f"  chunk_size={chunk_size}  overlap={overlap}  top_k={top_k}")
        print(f"{'='*60}\n")

    # ── Step 1: Extract text from the PDF ──
    if verbose:
        print("📖 Extracting text from PDF...")
    text = rag.extract_text(pdf_path)

    # ── Step 2: Chunk the text ──
    chunks = rag.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if verbose:
        print(f"✂️  Created {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")

    # ── Step 3: Embed all chunks ──
    if verbose:
        print("🧮 Embedding chunks (calling Gemini API)...")
    embeddings = rag.embed(chunks)

    # ── Step 4: Build FAISS index ──
    index = rag.build_index(embeddings)
    if verbose:
        print("🔍 FAISS index built\n")

    # ── Step 5: Ask each question ──
    correct = 0
    total = len(questions)
    results = []

    for i, q in enumerate(questions, 1):
        question = q["question"]
        expected = q["expected_keyword"].lower()

        if verbose:
            print(f"  Q{i:2d}: {question}")

        # Retrieve relevant chunks
        sources = rag.retrieve(question, index, chunks, k=top_k)

        # Generate answer
        answer = rag.generate_answer(question, sources)

        # Check if the expected keyword appears in the answer (case-insensitive)
        is_correct = expected in answer.lower()
        if is_correct:
            correct += 1

        results.append({
            "question": question,
            "expected_keyword": q["expected_keyword"],
            "answer": answer,
            "correct": is_correct,
        })

        if verbose:
            status = "✅" if is_correct else "❌"
            print(f"       {status}  Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            if not is_correct:
                print(f"       Expected keyword: '{q['expected_keyword']}'")
            print()

        # Small delay to respect rate limits
        time.sleep(0.5)

    # ── Print summary ──
    pct = (correct / total * 100) if total > 0 else 0
    if verbose:
        print(f"{'='*60}")
        print(f"  SCORE: {correct}/{total} correct ({pct:.0f}%)")
        print(f"{'='*60}")

    return correct, total, results


def run_all_configurations():
    """
    Run evaluation across multiple chunk_size and top_k configurations,
    then save a comparison table to RESULTS.md.
    """
    eval_dir = os.path.dirname(__file__)
    results_path = os.path.join(eval_dir, "RESULTS.md")

    chunk_sizes = [300, 500, 800]
    top_ks = [2, 4, 6]

    all_results = []

    print("\n🔬 Running evaluation across 9 configurations...\n")

    for cs in chunk_sizes:
        for tk in top_ks:
            print(f"\n{'─'*40}")
            print(f"  Configuration: chunk_size={cs}, top_k={tk}")
            print(f"{'─'*40}")

            correct, total, details = run_evaluation(
                chunk_size=cs,
                overlap=cs // 5,  # 20% overlap
                top_k=tk,
                verbose=True,
            )
            pct = (correct / total * 100) if total > 0 else 0
            all_results.append({
                "chunk_size": cs,
                "overlap": cs // 5,
                "top_k": tk,
                "correct": correct,
                "total": total,
                "percentage": pct,
                "details": details,
            })

            # Pause between configurations to respect rate limits
            print("  ⏳ Pausing 5s before next configuration...")
            time.sleep(5)

    # ── Find best configuration ──
    best = max(all_results, key=lambda r: (r["percentage"], r["top_k"]))

    # ── Write RESULTS.md ──
    md_lines = [
        "# DocuMind RAG Evaluation Results\n",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n",
        f"**Sample Document:** NovaTech Solutions Employee Handbook (fictional)\n",
        f"**Questions:** 15 factual questions with expected keyword matching\n",
        f"**Models:** Text={rag.TEXT_MODEL}, Embedding={rag.EMBEDDING_MODEL}\n",
        "",
        "## Configuration Comparison\n",
        "| Chunk Size | Overlap | Top-K | Score | Percentage |",
        "|:----------:|:-------:|:-----:|:-----:|:----------:|",
    ]

    for r in all_results:
        star = " ⭐" if r is best else ""
        md_lines.append(
            f"| {r['chunk_size']} | {r['overlap']} | {r['top_k']} | "
            f"{r['correct']}/{r['total']} | {r['percentage']:.0f}%{star} |"
        )

    md_lines.extend([
        "",
        f"## Best Configuration\n",
        f"- **Chunk Size:** {best['chunk_size']}",
        f"- **Overlap:** {best['overlap']}",
        f"- **Top-K:** {best['top_k']}",
        f"- **Score:** {best['correct']}/{best['total']} ({best['percentage']:.0f}%)\n",
        "",
        "## Analysis\n",
        "- **Smaller chunks (300)** tend to give more precise retrieval but may miss context",
        "  that spans across chunk boundaries.",
        "- **Larger chunks (800)** capture more context per chunk but may include irrelevant",
        "  information that dilutes the answer.",
        "- **Higher top_k** retrieves more passages, giving the LLM more context to work with,",
        "  but can also introduce noise.",
        "- The sweet spot balances precision (finding the right passage) with context",
        "  (giving the LLM enough surrounding information to answer well).",
        "",
        "## Per-Question Results (Best Configuration)\n",
        "| # | Question | Expected | Correct |",
        "|:-:|:---------|:---------|:-------:|",
    ])

    for i, d in enumerate(best["details"], 1):
        status = "✅" if d["correct"] else "❌"
        q_short = d["question"][:60] + ("..." if len(d["question"]) > 60 else "")
        md_lines.append(f"| {i} | {q_short} | {d['expected_keyword']} | {status} |")

    md_lines.append("")

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n📊 Results saved to: {results_path}")
    print(f"🏆 Best configuration: chunk_size={best['chunk_size']}, "
          f"top_k={best['top_k']} → {best['correct']}/{best['total']} ({best['percentage']:.0f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DocuMind RAG system")
    parser.add_argument("--chunk_size", type=int, default=500,
                        help="Character count per chunk (default: 500)")
    parser.add_argument("--top_k", type=int, default=4,
                        help="Number of chunks to retrieve (default: 4)")
    parser.add_argument("--overlap", type=int, default=None,
                        help="Overlap between chunks (default: chunk_size // 5)")
    parser.add_argument("--run_all", action="store_true",
                        help="Run all 9 configurations and save RESULTS.md")
    args = parser.parse_args()

    if args.run_all:
        run_all_configurations()
    else:
        overlap = args.overlap if args.overlap is not None else args.chunk_size // 5
        run_evaluation(
            chunk_size=args.chunk_size,
            overlap=overlap,
            top_k=args.top_k,
            verbose=True,
        )
