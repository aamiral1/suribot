# SuriBot — Grader Setup Guide

SuriBot is a Retrieval-Augmented Generation (RAG) chatbot system built for Suri Marketing, a digital marketing agency in Birmingham. It consists of a customer-facing chatbot widget and an admin dashboard. The admin can upload company documents and website content to the chatbot's knowledge base; the chatbot uses hybrid semantic + keyword search over that knowledge base to generate grounded, context-aware responses.

---

## Prerequisites

- **Python 3.10 or higher**
- **PostgreSQL 14 or higher** running locally on port 5432

### Installing PostgreSQL

- **macOS:** `brew install postgresql@14 && brew services start postgresql@14`
- **Ubuntu/Debian:** `sudo apt install postgresql && sudo systemctl start postgresql`
- **Windows:** Download and run the installer from https://www.postgresql.org/download/windows/

After installation, ensure a default database `postgres` exists with user `postgres`. The app expects the password to be `9999`. You can set this with:

```bash
psql -U postgres -c "ALTER USER postgres WITH PASSWORD '9999';"
```

If your local Postgres uses different credentials, you can override them via the `DB_*` variables in your `.env` file.

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Note: one dependency (`chunking_evaluation`) is installed directly from GitHub, so you need an internet connection and `git` available.

### 3. Configure environment variables

A pre-configured `.secret_env` file is provided with this submission (uploaded separately on Blackboard). Copy it to `.env`:

```bash
# macOS / Linux
cp .secret_env .env

# Windows
copy .secret_env .env
```

All API keys and credentials are already filled in — no further configuration needed. API credits have been pre-loaded on all services sufficient for testing purposes.

---

## Running the app

```bash
source venv/bin/activate   # Windows: venv\Scripts\activate
python app.py
```

The app will start on `http://localhost:5000` in debug mode. The database schema is created automatically on first run.

---

## Accessing the system

| Interface | URL |
|---|---|
| Chatbot | http://localhost:5000 |
| Admin dashboard | http://localhost:5000/admin |

**Admin login credentials:**
- Username: `admin`
- Password: `pass`

---

## What to try

**Chatbot (http://localhost:5000)**
- Open the chat widget and ask questions. If the knowledge base is empty the chatbot will still respond using its sales persona but without grounded context.

**Admin dashboard (http://localhost:5000/admin)**
- Upload a document under "Knowledge Base" to populate the knowledge base.
- Upload a website you would like crawled under sitemap parsing page.

---

## Architecture overview

- **Backend:** Flask + Flask-RESTful (Python)
- **Database:** PostgreSQL — stores document metadata, chunk data, and processing state
- **Document storage:** AWS S3 — stores raw uploaded files and extracted text
- **Vector database:** Pinecone (serverless) — hybrid sparse-dense index for RAG retrieval
- **AI/OCR:** OpenAI GPT-4o — text extraction from uploaded documents and chat response generation
- **Retrieval:** Hybrid search combining BM25 keyword scoring and OpenAI `text-embedding-3-small` semantic embeddings
- **Chatbot frontend:** Vanilla JS widget embedded in the Flask-served HTML page
