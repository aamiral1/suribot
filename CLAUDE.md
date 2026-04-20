# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context
We are building a be-spoke chatbot solution for a digital marketing agency based in Birmingham, called Suri Marketing. The solution comes with a chatbot (the front end) and an admin dashboard. The chatbot is intented to sit as a website widget on the client's website. The admin dashboard is the central hub to monitor and manage the deployed chatbot. The admin can view chatbot analytics (conversation metrics, KPIs, etc.), upload company documents to be added to the chatbot's knowledge base, provide a website domain for information to be extracted from that website's sitemap and also added to knowledge base.

## Planning workflow preference

When entering plan mode, always follow this workflow:

1. **Listen first** — let the user describe the problem and their proposed solution fully before asking anything.
2. **Ask clarifying questions** — use `AskUserQuestion` to resolve ambiguities or architectural choices before designing anything.
3. **Build a phased plan** — break the implementation into clearly named phases (e.g. Phase 1 — Database, Phase 2 — Extraction, etc.).
4. **Walk through phases one at a time** — present each phase with a plain-language explanation covering: what is being done, why, and any risks or tradeoffs. Do not present all phases at once.
5. **Get approval per phase** — wait for the user to approve, ask questions, or request changes before moving to the next phase. Update the plan file to reflect any agreed changes before moving on.
6. **Only call ExitPlanMode once all phases have been walked through and approved individually.**

When explaining a phase, write in plain English — avoid jargon, explain technical decisions and their tradeoffs, and invite challenge or alternatives at the end of each phase.

## Running the app

```bash
source venv/bin/activate
python app.py
```

The app runs on `http://localhost:5000` in debug mode. Admin dashboard at `/admin`.

## Environment variables

Required in `.env` (never commit this file):
- `OPENAI_API_KEY`
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_BUCKET`
- `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`

## Dependencies

Install from requirements (inside venv):
```bash
pip install -r requirements.txt
```

Notable non-PyPI dependency — chunking library installed directly from GitHub:
```
chunking_evaluation @ git+https://github.com/brandonstarxel/chunking_evaluation.git@e708410d1c61cb76a85cd9d433630ef89b9c6b85
```

## Infrastructure prerequisites

- **PostgreSQL** running locally on port 5432, database `postgres`, user `postgres`, password `9999`. Schema is auto-created on startup via `db.init_schema()`.
- **AWS S3** bucket (eu-north-1) — stores raw uploaded files, extracted text files, and the fitted BM25Encoder params (`bm25_encoder_params.json`).
- **Pinecone** serverless index (aws, us-east-1, **dotproduct** metric, 1536 dimensions) — sparse-dense hybrid index. If recreating, use `delete_and_recreate_index()` from `pinecone_store.py`.

## Architecture

### Document structure types

Every uploaded knowledge base document is tagged as either **Free Flow** or **Structured** at upload time (selected via the admin dashboard).

- **Free flow** — documents without clear section headings (e.g. a block of text, a policy written as one long piece). These are chunked semantically — the LLM decides where natural breaks are.
- **Structured** — documents with clear section headings (e.g. a brochure, an FAQ, a product guide). These are split at each heading into large "parent" sections, and each parent is then broken into smaller "child" pieces. The children are what gets searched; the parent is what gets shown to the LLM as context.

### How chunking works

**Free flow:** the full document text is passed to the semantic chunker, which produces a flat list of chunks. Each chunk is its own parent and child (no hierarchy).

**Structured:** the extracted text contains `[HEADING]` markers (added during OCR/extraction). The app splits the text at each heading to create parent sections, then runs the semantic chunker within each section to produce child chunks. Every child knows which parent section it belongs to.

Guardrail: if a section has fewer than 10 words, it gets merged into the next section. If more than half the sections needed merging, a warning is logged — it usually means the heading detection wasn't reliable for that document.

### How search works (hybrid retrieval)

When a user asks a question, the system searches the knowledge base in two ways simultaneously:

1. **Semantic search** — the question is converted into a vector (a list of numbers that captures meaning) and compared against all stored chunks to find semantically similar content.
2. **Keyword search** — a BM25 scorer finds chunks that contain the actual words from the question.

Both searches happen in a single Pinecone query. The results are blended using an alpha value (default 0.5 = equal weight). The admin can adjust this balance using a slider in the dashboard. A higher alpha favours semantic matching; lower favours keyword matching.

The top 2 results (by combined score) are used. For structured documents, the full parent section is retrieved and given to the LLM as context — not just the small matched child chunk. This gives the LLM broader context to answer from.

### Chat flow

Every user message to `/api`:
1. Retrieves the top 2 relevant parent sections from the knowledge base.
2. Builds the LLM call as:
   - **System message:** the sales persona/instructions from admin-uploaded system prompt documents.
   - **User message:** retrieved KB sections + instructions on how to use them + the user's question.
3. If the KB is empty, sends the message directly without context.

### Key decisions and things to know

- **`document_chunks` table** — a Postgres table that stores every chunk (parent text + child text) for every KB document. Used to look up parent sections at query time, and to rebuild the keyword search model after restarts.
- **BM25 keyword model** — learns word patterns from all KB chunks. Saved to S3 as `bm25_encoder_params.json` after every ingestion. Loaded automatically on startup.
- **Pinecone index** — uses `metric=dotproduct` (not cosine) to support hybrid search. If recreating the index, use `delete_and_recreate_index()` from `pinecone_store.py`. Never change back to cosine or hybrid search will break.
- **System prompt cache** — rebuilt from S3 on every startup. The full text is prepended to every LLM call as the system message. GPT has no memory between calls so this must be sent every time.
- **Document types:** `knowledge_base` documents go through the chunking/embedding pipeline. `system_prompt` documents are LLM-transformed and stored as the chatbot's persona — they never get chunked or searched.
- The DB enum types are guarded so `init_schema()` is safe to call on every startup without errors.
- `AllowedFileTypes`: PDF, DOCX, TXT, MD, PNG. JPEG/JPG are handled in extraction but not in the enum — both need updating if adding JPEG support.