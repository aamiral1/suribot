from flask import Flask, render_template, jsonify, request
from flask_restful import Api, Resource, reqparse
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from dotenv import load_dotenv
from openai import OpenAI
from werkzeug.utils import secure_filename
from database import Database
from doc_parser import extract_doc_info
from chunker import chunk_text, chunk_text_recursive, chunk_text_fixed
from config import ChunkingStrategy, get_chunking_strategy
from embedder import embed_chunks
from pinecone_store import get_or_create_index, upsert_chunks, hybrid_query, query_index
from structured_chunker import split_into_sections, chunk_sections_with_fallback
from hybrid_retriever import hybrid_retrieve
import bm25_encoder as bm25_enc
from enums import DocumentStatus, SourceType, AllowedFileTypes
import boto3
import botocore
import threading
import custom_exceptions as ex
import datetime
import os
import tempfile

load_dotenv()

SYSTEM_PROMPT_TRANSFORM_PROMPT = """You are a system prompt transformation engine.

Your task is to convert any uploaded system prompt document into a clean, structured, and production-ready system prompt for an LLM or chatbot.

----------------------------------------
CORE OBJECTIVE
----------------------------------------

- Preserve the original meaning, intent, and details as accurately as possible
- Improve structure, clarity, and usability for an LLM
- Do NOT introduce new information, assumptions, or capabilities

----------------------------------------
INPUT CHARACTERISTICS
----------------------------------------

The uploaded document may contain:
- Instructions
- Notes or drafts
- Business logic
- Service descriptions
- Policies or constraints
- Conversation examples
- Unstructured or messy content

You must handle all formats adaptively.

----------------------------------------
TRANSFORMATION RULES
----------------------------------------

1. Extract only meaningful instructional content
2. Preserve all important details exactly where possible (facts, rules, constraints, workflows)
3. Remove:
   - Redundancy
   - Repetition
   - Filler or informal notes
4. Rewrite unclear or messy phrasing into precise, directive instructions
5. Infer structure ONLY (not content) where needed to improve clarity

----------------------------------------
STRICT CONSTRAINTS
----------------------------------------

- Do NOT hallucinate missing information
- Do NOT invent services, rules, or capabilities
- Do NOT generalise beyond what is stated
- Do NOT change meaning, even if the wording is improved
- Do NOT omit important constraints or edge cases
- If something is ambiguous, keep it as-is but clarify wording without adding meaning

----------------------------------------
OUTPUT STRUCTURE
----------------------------------------

Produce ONE final system prompt with clear sections such as:

- Role / Identity
- Objectives / Responsibilities
- Knowledge / Domain (if present)
- Processes / Workflows (if present)
- Behaviour / Communication Style (if present)
- Rules / Constraints / Guardrails
- Task-specific instructions (if present)

Only include sections that are supported by the input document.

----------------------------------------
STYLE GUIDELINES
----------------------------------------

- Use clear, directive language suitable for LLM execution
- Keep wording concise and precise
- Avoid conversational or explanatory tone
- Write in a way that can be directly pasted into a system prompt field

----------------------------------------
OUTPUT RULES
----------------------------------------

- Output ONLY the final transformed system prompt
- Do NOT include explanations, commentary, or analysis
- Do NOT reference the source document
- Do NOT include placeholders unless they exist in the original

----------------------------------------
FAIL-SAFE BEHAVIOUR
----------------------------------------

If the document is:
- Incomplete → preserve what exists without filling gaps
- Messy → organise it without adding meaning
- Repetitive → deduplicate while preserving all unique information

Your goal is fidelity first, structure second, improvement third."""

app = Flask(__name__)
app.config["SECRET_KEY"] = "supersecretkey"
app.config["UPLOAD_FOLDER"] = "static/files"
app.config["OCR_PROCESSING_FOLDER"] = "static/ocr"
app.config["EXTRACTED_TEXT_FOLDER"] = "static/extracted_text"
api = Api(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=20)

response = client

# Initialise Postgres Database
db = Database("document_database.db", "documents")
db.init_schema()

# In-memory system prompt cache — rebuilt on startup and after each system prompt doc is added to KB
system_prompt_cache = ""

# Retrieval alpha — blend between semantic (1.0) and keyword (0.0), default equal weight
retrieval_alpha = 0.5

def _rebuild_system_prompt_cache():
    global system_prompt_cache
    parts = []
    for s3_bucket, s3_key in db.get_system_prompt_docs():
        parts.append(_get_file_text(s3, s3_bucket, s3_key))
    system_prompt_cache = "\n\n".join(parts).strip()
    print(f"[Cache] System prompt cache rebuilt ({len(system_prompt_cache)} chars).")
    print(system_prompt_cache)

# Initialise AWS S3 Client
s3 = boto3.resource(
    service_name="s3",
    region_name="eu-north-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

bucket_name = os.getenv("AWS_S3_BUCKET")


@app.route("/")
def home():
    return render_template("index.html")


# REST API for backend
chatbot_args = reqparse.RequestParser()
chatbot_args.add_argument("message", type=str, help="Message for LLM", required=True)


# API that handles communication with Open AI
class ChatbotAPI(Resource):
    def post(self):
        args = chatbot_args.parse_args()
        user_message = args["message"]

        messages = []

        SYSTEM_PROMPT = """You are the assistant for Suri Marketing, a social media marketing agency. You speak like a warm, knowledgeable member of their team — friendly, confident, not pushy, never robotic.
        ## Scope
        - You only help with questions about Suri Marketing: our services, packages, pricing, process, how we work with clients.
        - If someone asks you to do something unrelated (write essays, translate, code, general advice, anything off-topic), politely redirect: "I'm here to help with questions about Suri Marketing — happy to tell you about what we offer if that's useful."
        - Never mention being an AI, a language model, or that you have a "knowledge base" or "context." You're just Suri Marketing's assistant.

        ## How to answer
        - Every factual claim (prices, package details, services, policies) must come from information given to you in this conversation. If a fact isn't there, say so naturally: "I don't have that detail handy — the best person to answer that is someone from our team. Want me to point you to them?"
        - Do NOT invent facts. But DO rephrase what you're given into natural conversational language. Never paste chunks verbatim. Think: how would a sales rep say this out loud?
        - Lead with what the customer cares about (value, fit, outcome), then details.
        - Keep replies short and scannable. Use bullets only when genuinely listing 3+ parallel items. Default to prose.
        - If someone's rude or inappropriate, stay polite and steer back: "I'm here to help with Suri Marketing questions — let me know if there's something I can help with."

        ## Examples of good style
        User: What services do you offer?
        You: We run social media marketing campaigns end-to-end — content strategy, video creation, posting and scheduling across Instagram, TikTok, and Facebook, plus profile optimisation and caption writing. We have two packages depending on how much content you want each month (10 or 14 videos), both starting from a 4-month campaign. Want me to walk you through the pricing?

        User: How much does it cost?
        You: Our packages start at £995/month for 10 videos across your social channels, or £1195/month for 14 videos — both include the full service: strategy, filming guidance, editing, captions, scheduling, the lot. If you'd rather pay upfront, you save £100/month on either one (so £895 or £1095). Do you have a rough sense of how much content you're wanting each week? That usually makes it easier to figure out which one fits.

        User: What's the difference between Package A and B?
        You: Honestly the main difference is volume — Package A gets you 10 videos every 4 weeks, Package B gets you 14. Everything else is the same: same strategy work, same platforms (Instagram, TikTok, Facebook), same editing and posting. Package B works well if you're trying to grow fast or post most days; Package A is plenty if you're building steadily. What are you hoping to achieve with your socials?

        User: Why is the minimum 4 months?
        You: Social media takes a bit of runway to actually work — the first month or two is mostly building up content, testing what lands with your audience, and letting the platforms figure out who to show you to. Four months gives us enough time to show you real results rather than just posts going out. Most clients start seeing momentum around month two or three.

        User: Do you do Google Ads?
        You: That's not something we handle — we focus on organic social media content across Instagram, TikTok, and Facebook rather than paid ads. If content and community-building on those platforms is useful for you though, I'd love to tell you more about how we work.

        User: Can you write me a caption for my post about coffee?
        You: I'm here to chat about how Suri Marketing could help with your social media — writing captions is actually part of what we do for our clients as part of a full content package. Happy to tell you more about that if it's useful?

        User: What's your refund policy?
        You: I don't have that detail handy — that's best answered by someone on our team directly. Want me to point you their way?
        """

        messages.append({"role": "system", "content": SYSTEM_PROMPT})

        # if system_prompt_cache:
        #     messages.append({"role": "system", "content": system_prompt_cache})


        # Retrieve relevant chunks from the knowledge base
        context_block = ""
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index_name = os.getenv("PINECONE_INDEX_NAME")
            if index_name in [i.name for i in pc.list_indexes()]:
                pinecone_index = pc.Index(index_name)
                retrieved = hybrid_retrieve(
                    query=user_message,
                    client=client,
                    pinecone_index=pinecone_index,
                    top_k=4,
                    alpha=retrieval_alpha,
                )
                if retrieved:
                    print(f"\n[/api] Retrieved {len(retrieved)} chunk(s) for query: '{user_message}'")
                    for i, r in enumerate(retrieved, 1):
                        print(f"  [{i}] {r['text']}")
                    context_block = "\n\n---\n\n".join(r["text"] for r in retrieved)
        except Exception as e:
            print(f"[ChatbotAPI] RAG retrieval failed, continuing without context: {e}")

        if context_block:
            messages.append({
            "role": "system",
            "content": (
                "Relevant information for the user's current question "
                "(use this as your source of truth, but rephrase naturally — "
                "do not paste it verbatim):\n\n"
                f"{context_block}"
            )
        })

        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages
        )

        return {"response": response.choices[0].message.content}, 200


api.add_resource(ChatbotAPI, "/api")


# REST API to poll document status (status, has_text)
class DocumentStatusAPI(Resource):
    def get(self, doc_id):

        # get status and extracted file path if available
        status = db.get_status(doc_id)

        if status == DocumentStatus.PROCESSING:
            msg = {"status": "processing", "has_text": False}
            resp = jsonify(msg)
            resp.status_code = 200

            return resp

        elif status == DocumentStatus.CREATED:
            msg = {"status": "created", "has_text": False}
            resp = jsonify(msg)
            resp.status_code = 200

            return resp

        elif status == DocumentStatus.FAILED:
            msg = {"status": "failed", "has_text": False}
            resp = jsonify(msg)
            resp.status_code = 200

            return resp

        elif status == DocumentStatus.SUCCESS:
            # get extracted text s3 bucket and key of document
            try:
                s3_bucket, s3_key = db.get_extracted_text_file_path(doc_id)

            except Exception as e:

                return jsonify(
                    {
                        "status": "error",
                        "has_text": False,
                        "message": "DB Error: " + str(e),
                    }
                )

            # check if file exists in S3
            if _file_exists(s3, s3_bucket, s3_key):
                msg = {"status": "success", "has_text": True}
                resp = jsonify(msg)
                resp.status_code = 200

                return resp
            else:
                msg = {"status": "error", "has_text": False}
                resp = jsonify(msg)
                resp.status_code = 200

                return resp

        else:
            # generic error
            msg = {"status": "failed", "has_text": False}
            resp = jsonify(msg)
            resp.status_code = 200

            return resp


api.add_resource(DocumentStatusAPI, "/document/<string:doc_id>/status")


# REST API to get extracted text from document
class ExtractedDocumentTextAPI(Resource):
    def get(self, doc_id):
        # get extracted text file path of document
        try:
            s3_bucket, s3_key = db.get_extracted_text_file_path(doc_id)

        except Exception as e:
            return jsonify({"text": None})

        text = None
        try:
            if _file_exists(s3, s3_bucket, s3_key):
                text = _get_file_text(s3, s3_bucket, s3_key)
        except Exception as e:
            print(f"[ExtractedTextAPI] S3 error for {doc_id}: {e}")

        return jsonify({"text": text})


api.add_resource(ExtractedDocumentTextAPI, "/document/<string:doc_id>/text")


# REST API to get all documents
@app.route("/documents", methods=["GET"])
def get_documents():
    try:
        rows = db.get_all_documents()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    documents = [
        {
            "doc_id":        row[0],
            "source_type":   row[1],
            "file_name":     row[2],
            "file_size":     row[3],
            "file_type":     row[4],
            "uploaded_date": str(row[5]),
            "status":        row[8],
            "in_kb":         row[12],
        }
        for row in rows
    ]

    return jsonify({"documents": documents}), 200


# REST API to mark a document as added to the knowledge base
@app.route("/document/<string:doc_id>/add-to-kb", methods=["POST"])
def add_to_kb(doc_id):
    doc_type = db.get_doc_type(doc_id)
    db.set_kb_status(doc_id, "processing")
    if doc_type == "system_prompt":
        threading.Thread(target=_system_prompt_job, args=(doc_id,)).start()
    else:
        threading.Thread(target=_chunk_and_embed_job, args=(doc_id,)).start()
    return jsonify({"status": "processing"}), 200


@app.route("/document/<string:doc_id>/kb-status", methods=["GET"])
def get_kb_status(doc_id):
    kb_status = db.get_kb_status(doc_id)
    return jsonify({"kb_status": kb_status}), 200


def _system_prompt_job(doc_id):
    print(f"\n[KB] Starting system prompt job for doc: {doc_id}")
    try:
        # fetch extracted text from S3
        s3_bucket, s3_key = db.get_extracted_text_file_path(doc_id)
        extracted_text = _get_file_text(s3, s3_bucket, s3_key)

        # transform into structured system prompt via LLM
        print(f"[KB] Transforming system prompt doc ({len(extracted_text)} chars)...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TRANSFORM_PROMPT},
                {"role": "user", "content": extracted_text},
            ],
        )
        transformed = response.choices[0].message.content.strip()
        print(f"[KB] Transformation complete ({len(transformed)} chars).")

        # overwrite S3 extracted text file with transformed version
        s3.meta.client.put_object(
            Bucket=s3_bucket,
            Key=s3_key,
            Body=transformed.encode("utf-8"),
            ContentType="text/plain",
        )

        db.set_in_kb(doc_id)
        db.set_kb_status(doc_id, "success")
        _rebuild_system_prompt_cache()
        print(f"[KB] System prompt doc {doc_id} marked as in_kb=TRUE.")
    except Exception as e:
        print(f"[KB] Error during system prompt job: {e}")
        db.set_kb_status(doc_id, "failed")


def _chunk_and_embed_job(doc_id):
    strategy = get_chunking_strategy()
    print(f"\n[KB] Starting chunk + embed job for doc: {doc_id} (strategy: {strategy.value})")
    try:
        # fetch extracted text from S3
        s3_bucket, s3_key = db.get_extracted_text_file_path(doc_id)
        text = _get_file_text(s3, s3_bucket, s3_key)

        # chunk using selected strategy
        print(f"[KB] Chunking text ({len(text)} chars)...")
        if strategy == ChunkingStrategy.SECTION_AND_SEMANTIC:
            chunks = chunk_sections_with_fallback(text, api_key=os.getenv("OPENAI_API_KEY"))
        elif strategy == ChunkingStrategy.RECURSIVE_TOKEN:
            chunks = chunk_text_recursive(text)
        elif strategy == ChunkingStrategy.FIXED_SIZE:
            chunks = chunk_text_fixed(text)
        else:  # SEMANTIC_ONLY
            chunks = chunk_text(text, api_key=os.getenv("OPENAI_API_KEY"))
        print(f"[KB] Produced {len(chunks)} chunks.")

        # embed (dense)
        embedded_chunks = embed_chunks(client, chunks)

        # build chunk rows for document_chunks
        chunk_rows = [
            {
                "chunk_id": c["chunk_index"],
                "text": c["text"],
                "heading": None,
            }
            for c in embedded_chunks
        ]

        # encode sparse vectors
        encoder = bm25_enc.get_encoder()
        texts = [c["text"] for c in embedded_chunks]
        if encoder is not None:
            sparse_vectors = bm25_enc.encode_documents(encoder, texts)
        else:
            encoder = bm25_enc.fit_and_save(texts, s3, bucket_name)
            sparse_vectors = bm25_enc.encode_documents(encoder, texts)

        # build format expected by upsert_chunks
        chunks_for_upsert = [
            {
                "chunk_id": c["chunk_index"],
                "text": c["text"],
                "embedding": c["embedding"],
            }
            for c in embedded_chunks
        ]

        # upsert to Pinecone with dense + sparse vectors
        print(f"[KB] Upserting {len(chunks_for_upsert)} hybrid vectors to Pinecone...")
        pinecone_index = get_or_create_index(
            api_key=os.getenv("PINECONE_API_KEY"),
            index_name=os.getenv("PINECONE_INDEX_NAME"),
        )
        upsert_chunks(pinecone_index, doc_id, chunks_for_upsert, sparse_vectors)

        # persist chunk rows to document_chunks table
        db.insert_chunks(doc_id, chunk_rows)
        print(f"[KB] Inserted {len(chunk_rows)} rows into document_chunks.")

        # refit and save BM25 encoder with updated corpus
        all_chunks = db.get_all_chunks()
        bm25_enc.fit_and_save(
            [c["text"] for c in all_chunks],
            s3,
            bucket_name,
        )

        # mark document as in KB
        db.set_in_kb(doc_id)
        db.set_kb_status(doc_id, "success")
        print(f"[KB] Document {doc_id} marked as in_kb=TRUE in database.")

    except Exception as e:
        print(f"[KB] Error during chunk + embed job: {e}")
        db.set_kb_status(doc_id, "failed")


@app.route("/query", methods=["POST"])
def query_kb():
    data = request.get_json()
    query = data["query"]

    from pinecone import Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if index_name not in [i.name for i in pc.list_indexes()]:
        print("No vector database available")
        return jsonify({"results": []}), 200

    pinecone_index = pc.Index(index_name)
    results = hybrid_retrieve(
        query=query,
        client=client,
        pinecone_index=pinecone_index,
        top_k=2,
        alpha=retrieval_alpha,
    )
    return jsonify({"results": results}), 200


@app.route("/test-rag", methods=["POST"])
def test_rag():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400

    from pinecone import Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if index_name not in [i.name for i in pc.list_indexes()]:
        return jsonify({"error": "Pinecone index not found"}), 500

    pinecone_index = pc.Index(index_name)
    results = hybrid_retrieve(
        query=query,
        client=client,
        pinecone_index=pinecone_index,
        top_k=2,
        alpha=retrieval_alpha,
    )

    return jsonify({
        "query": query,
        "alpha": retrieval_alpha,
        "results": [
            {
                "rank": i + 1,
                "score": round(r["score"], 4),
                "doc_id": r["doc_id"],
                "chunk_id": r["chunk_id"],
                "text": r["text"],
            }
            for i, r in enumerate(results)
        ],
    }), 200


@app.route("/config/alpha", methods=["POST"])
def set_alpha():
    global retrieval_alpha
    data = request.get_json()
    alpha = data.get("alpha")
    if alpha is None or not isinstance(alpha, (int, float)) or not (0.0 <= alpha <= 1.0):
        return jsonify({"error": "alpha must be a number between 0.0 and 1.0"}), 400
    retrieval_alpha = float(alpha)
    print(f"[config] retrieval_alpha updated to {retrieval_alpha}")
    return jsonify({"alpha": retrieval_alpha}), 200


# Admin Dashboard
class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload File")


@app.route("/admin", methods=["GET", "POST"])
def admin():
    form = UploadFileForm()

    if request.method == "GET":
        return render_template("admin.html", form=form)

    if form.validate_on_submit():
        file = form.file.data  # get the uploaded file data sent from frontend
        file_name = secure_filename(file.filename)
        file_size = str(_format_size(_get_file_size(file)))
        file_type = AllowedFileTypes.from_filename(file_name)

        # check if file already exists in S3 bucket
        if _file_exists(s3, bucket_name, file_name):
            print("File already exits")

            msg = {"status": "fail", "error": "File already exists"}

            return jsonify(msg), 400

        # upload file to S3 storage
        s3.meta.client.upload_fileobj(file, bucket_name, file_name)

        # Register file entry in database
        doc_type = request.form.get("doc_type", "knowledge_base")
        doc_id = db.create(
            source_type=SourceType.UPLOAD.value,
            name=file_name,
            size=file_size,
            type=file_type.value,
            upload_date=datetime.datetime.now(),
            s3_file_bucket=bucket_name,
            s3_file_key=file_name,
            s3_extracted_text_bucket="na",
            s3_extracted_text_key="na",
            doc_type=doc_type,
            doc_structure="structured",
        )

        msg = {
            "status": "ok",
            "doc_id": doc_id,
            "s3_file_bucket": bucket_name,
            "s3_file_key": file_name,
        }

        print("File has been uploaded succesfully.")
        print(f"Current status of file is {db.get_status(doc_id)}")

        return jsonify(msg), 200

    else:
        return jsonify({"status": "fail", "error": "validation failed"}), 400


# Route to extract text from an uploaded document
@app.route("/extract-text", methods=["POST"])
def extract_text():
    # wrapper function to thread extraction: Replace with Celery later

    # AWS S3 - CONTINUE FROM HERE!
    def extraction_job(doc_id, file_path):
        file_name = doc_id + ".txt"
        isSuccess = False

        try:
            extracted_text = extract_doc_info(
                client=client,
                file_path=file_path,
            )

            # upload extracted text as file to S3 bucket
            s3.meta.client.put_object(
                Bucket=os.getenv("AWS_S3_BUCKET"),
                Key=secure_filename(file_name),
                Body=extracted_text.encode("utf-8"),
                ContentType="text/plain",
            )

            # register extracted text file path in database
            try:
                db.set_extraction_text_path(
                    doc_id=doc_id,
                    s3_extracted_text_bucket=bucket_name,
                    s3_extracted_text_key=secure_filename(file_name),
                )

                isSuccess = True

            except Exception as e:
                print("Failed to update extracted file path of document")
                raise

            try:
                # update document status based on whether extracted text file exists
                if isSuccess:
                    db.transition_status(
                        doc_id=doc_id, new_status=DocumentStatus.SUCCESS
                    )
                else:
                    db.transition_status(
                        doc_id=doc_id, new_status=DocumentStatus.FAILED
                    )

            except ex.InvalidDocumentStatusTransition as e:
                db.transition_status(doc_id=doc_id, new_status=DocumentStatus.FAILED)
                raise

            except Exception as e:
                db.transition_status(doc_id=doc_id, new_status=DocumentStatus.FAILED)
                raise

        except ex.InvalidDocumentStatusTransition as e:
            db.transition_status(doc_id=doc_id, new_status=DocumentStatus.FAILED)
            print(e)

        except ex.ExtractionTimeOut as e:
            db.transition_status(doc_id=doc_id, new_status=DocumentStatus.FAILED)
            print(e)

        except Exception as e:
            db.transition_status(doc_id=doc_id, new_status=DocumentStatus.FAILED)
            print(f"Error: {e}")

        finally:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)

    # get doc id from POST request
    data = request.get_json()
    print(data)
    doc_id = data["doc_id"]

    # Check if document is valid to be processed
    doc_status = db.get_status(doc_id)

    if doc_status == DocumentStatus.CREATED or doc_status == DocumentStatus.FAILED:
        # update doc status to Processing and perform extraction
        try:
            db.transition_status(
                doc_id=doc_id, new_status=DocumentStatus.PROCESSING
            )  # set doc status to processing

            # extract file path from doc_id based on DB
            s3_bucket, s3_key = db.get_file_path(doc_id)
            local_file_path = _download_s3_file_to_temp(s3, s3_bucket, s3_key)

            threading.Thread(
                target=extraction_job, args=(doc_id, local_file_path)
            ).start()

        except ex.InvalidDocumentStatusTransition as e:
            msg = {
                "status": "failed",
                "message": "Unable to update document status to PROCESSING on database (IDST)",
            }

            return jsonify(msg), 500

        except Exception as e:
            msg = {"status": "failed", "message": str(e)}
            db.transition_status(doc_id=doc_id, new_status=DocumentStatus.FAILED)
            print(f"Error: {e}")
            return jsonify(msg), 500


        msg = {"status": "began processing"}

        return jsonify(msg), 200

    elif doc_status == DocumentStatus.PROCESSING:
        msg = {"status": "already processing"}
        return jsonify(msg), 200

    elif doc_status == DocumentStatus.SUCCESS:
        msg = {"status": "already extracted"}
        return jsonify(msg), 200

    else:
        msg = {"status": "error"}
        return jsonify(msg), 500



# AWS S3 Helper Functions
def _file_exists(resource, bucket, key):
    try:
        resource.meta.client.head_object(Bucket=bucket, Key=key)
        print(f"File: '{key}' found!")
        return True
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print(f"File'{key}' File does not exist!")
            return False
        else:
            print("Something else went wrong")
            return False


def _get_file_text(resource, bucket, key):
    # get the object
    response = resource.meta.client.get_object(Bucket=bucket, Key=key)

    # Read the file contents
    file_content = response["Body"].read()

    return file_content.decode("utf-8")


def _get_file_size(file):
    pos = file.stream.tell()
    file.stream.seek(0, 2)
    size = file.stream.tell()
    file.stream.seek(pos)
    return size


def _format_size(size_bytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024


def _get_file_type(filename: str) -> AllowedFileTypes:
    if not filename or "." not in filename:
        raise ex.InvalidFileType("Invalid file name")

    ext = os.path.splitext(filename)[1].lower().strip(".")  # 'pdf'
    ext_upper = ext.upper()  # 'PDF'

    try:
        return AllowedFileTypes(ext_upper)
    except ValueError:
        raise ex.InvalidFileType(f"Unsupported file type: {ext_upper}")

def _download_s3_file_to_temp(s3_resource, bucket, key):
    suffix = os.path.splitext(key)[1] or ".tmp"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()

    s3_resource.meta.client.download_file(bucket, key, tmp_path)
    return tmp_path


# Build system prompt cache from any system prompt docs already in KB
_rebuild_system_prompt_cache()

# Load BM25 encoder from S3 if available; otherwise fit from existing chunks
_loaded_encoder = bm25_enc.load_from_s3(s3, bucket_name)
if _loaded_encoder is None:
    existing_chunks = db.get_all_chunks()
    if existing_chunks:
        bm25_enc.fit_and_save(
            [c["text"] for c in existing_chunks],
            s3,
            bucket_name,
        )

if __name__ == "__main__":
    app.run(debug=True)
