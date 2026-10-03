# Advanced RAG Pipeline

A production-grade Retrieval-Augmented Generation system built with **LangGraph**, **LangChain**, and **Groq** — adaptive query rewriting, web search fallback, RAPTOR-style hierarchical indexing, and self-correction loops for hallucination and answer sufficiency.

Built as a 10-day project, backend-only. Started from a hand-drawn system design before any code was written.

## What this is

Most RAG demos are a single retrieve-then-generate chain. This project routes each question down one of three paths (clarify, index lookup, or web search), picks a query-rewriting strategy based on the *kind* of question it is rather than trying all of them, and checks its own output for grounding and completeness before returning an answer — regenerating or re-retrieving when it fails either check, capped so it can never loop forever.

## Architecture

```
User question
      |
Query router (LLM classifies: clarify / index_lookup / web_search)
      |
   +--+--+------------------+
   |     |                  |
clarify  index_lookup   web_search
   |     |                  |
  END  strategy         search-query
      selector           rewrite
        |                  |
    retrieval           Tavily
        |                  |
        +--- grade: relevant? ---+
                  | (retry -> re-select / re-search, capped)
                  v
            generate_answer
                  |
          grade: grounded in docs? --(retry, capped)--> regenerate
                  |
          grade: answers the question? --(retry, capped)--> re_retrieve --> generate_answer
                  |
                 END
```

Indexing (separate from the query-time graph above): PDF -> chunks -> Chroma. Optionally extended with RAPTOR (hierarchical cluster summaries) and multi-query aliases (paraphrased embeddings pointing back to the source chunk). All node types live in **one** Chroma collection — retrieval doesn't select a method, a single `similarity_search` returns whichever nodes are nearest, of any type.

## Quick start

1. Install dependencies (uncomment and run the install cell once):
   ```
   pip install langgraph langchain-groq langchain-community langchain-chroma \
       langchain-huggingface langchain-text-splitters chromadb sentence-transformers \
       tavily-python scikit-learn python-dotenv
   ```
2. Create a `.env` file in the project directory:
   ```
   GROQ_API_KEY=your_key_here
   TAVILY_API_KEY=your_key_here
   LANGCHAIN_API_KEY=your_key_here
   ```
3. Place a source PDF at `./sample.pdf` (or change `SOURCE_PATH`).
4. Run the notebook top to bottom.

First run downloads the embedding model (~130MB, cached after) and builds the Chroma index — expect this to take a minute or two longer than subsequent runs.

## Configuration

| Setting | Where | Purpose |
|---|---|---|
| `USE_ADVANCED_INDEXING` | config cell | `False` skips RAPTOR/multi-query (no extra LLM calls at index time), `True` builds both |
| `MAX_RETRIES` | config cell | Cap on every self-correction loop (strategy retry, hallucination retry, sufficiency retry) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | config cell | Currently 500/50, generic default — tune per-document once domain is fixed |
| `MODEL` | config cell | `openai/gpt-oss-120b` via Groq; any Groq model with tool-calling support works |

## Key design decisions

Worth reading if you're extending this or reviewing it for a portfolio/interview — these were deliberate trade-offs, not oversights:

- **Single-strategy classification, not exhaustive cycling.** The LLM picks *one* query-rewriting strategy (RAG-Fusion, Decomposition, Step-back, HyDE, or plain retrieval) based on the question type, matched to the use-case hints in the original system design. A failed grade re-classifies excluding what already failed, rather than round-robining blindly.
- **RAPTOR uses KMeans (hard clustering), not the original paper's UMAP + Gaussian Mixture Models (soft clustering).** Reasonable simplification for a 10-day timeline; a query can't match a chunk that's split across two clusters as cleanly as the full paper's approach would allow.
- **ColBERT was scoped out entirely**, not simplified. It requires token-level multi-vector matching, which is a fundamentally different retrieval mechanism — it can't be added as "more text to embed" into the shared Chroma collection the way RAPTOR and multi-query indexing were.
- **Two independent graders after generation**, not one combined check: hallucination (is the answer grounded in the *retrieved documents*?) and sufficiency (does it *answer the question*?). An answer can pass one and fail the other — keeping them separate lets each retry target the actual problem instead of guessing.
- **`resolve_content()` is the one function every retrieval path must go through.** Multi-query alias nodes embed a paraphrase, not the real content — this function is what prevents a paraphrase from leaking into the LLM's context as if it were source text.

## Known limitations / future work

- ColBERT indexing (see above — would need a separate retriever, e.g. via RAGatouille, and logic to merge/rerank across two retrieval systems)
- UMAP + GMM soft clustering for RAPTOR, replacing the current KMeans simplification
- `re_retrieve` currently always falls back to query decomposition; could instead route based on *why* the original attempt failed
- No frontend/API layer yet — backend only, by original scope
- Indexing methods run additively into one collection with no way to weight or prefer one type over another at retrieval time

## Extending this

Indexing methods are decoupled from retrieval by design — any method whose output is "text to embed, optionally tagged with a pointer back to real content" (`node_type` + `source_chunk` metadata) plugs into the same collection with no changes to the retriever, `resolve_content`, or the graph. Removing RAPTOR or multi-query indexing is a two-line change (delete the `.build()`/`.index()` call). Adding a similarly-shaped method costs one new class. Adding a method with its own retrieval mechanism (like ColBERT) is a bigger lift — a second retriever plus merge/rerank logic.