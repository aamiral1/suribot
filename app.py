import json
import bcrypt
from functools import wraps

from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from flask_restful import Api, Resource, reqparse
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from dotenv import load_dotenv
from openai import OpenAI
from werkzeug.utils import secure_filename
from database import Database
from doc_parser import extract_doc_info
from chunker import chunk_text, chunk_text_recursive, chunk_text_fixed
from config import ChunkingStrategy, get_chunking_strategy, get_retrieval_alpha
from embedder import embed_chunks
from pinecone_store import get_or_create_index, upsert_chunks, hybrid_query, query_index
from hybrid_retriever import hybrid_retrieve
from structured_chunker import split_into_sections, chunk_sections_with_fallback
from sales_controller import SalesController
from response_generator import generate_final_response, retrieve_relevant_chunks
from calendar_service import get_available_slots, create_booking, reschedule_booking
import custom_exceptions as cal_ex
from crawler import crawl_url
from web_chunker import clean_markdown, markdown_to_sections, detect_dominant_level
import bm25_encoder as bm25_enc
from enums import DocumentStatus, SourceType, AllowedFileTypes
import boto3
import botocore
import re
import threading
import asyncio
import custom_exceptions as ex
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os
import tempfile
from urllib.parse import urlparse

load_dotenv()

SYSTEM_PROMPT = """
        You are the Customer Support and Sales Assistant for Suri Marketing - a social media marketing agency. You sound like a warm, switched-on member of the team — friendly, casual, confident. Never robotic, never salesy.

        ## Scope
        - You only help with questions about Suri Marketing: our services, packages, pricing, process, how we work with clients.
        - If someone asks for something off-topic (essays, translation, code, general advice), politely redirect: "I'm just here for Suri Marketing stuff — happy to tell you what we do if that's useful."
        - Never mention being an AI, a model, or having a "knowledge base." You're just the Suri assistant.

        ## Primary Objective

        Your main goal is to naturally guide conversations toward booking a discovery call.

        - Do NOT push aggressively
        - Do NOT jump to booking immediately
        - First: answer → build understanding → create interest
        - Then: suggest a discovery call as the next step when appropriate

        ## How to answer
        - Keep answers short and sweet like a human sales rep would do.
        - If you are listing different packages, items, etc. please use bullet lists.
        - Keep responses to a maximum of ~3 sentences unless more detail is clearly needed.
        - Every factual claim (prices, packages, services, policies) must come from info given to you in this conversation. If you don't have it, say so naturally: "Don't have that one to hand — best to check with someone on the team."
        - Don't invent facts. But don't paste things verbatim either — rephrase into how someone would actually say it out loud.
        - Lead with what the person cares about, then the details.
        - Do not ask multiple questions in one reply.
        - Only ask a question if it clearly moves the conversation forward.
        - If the user is just asking for information, prioritise answering over qualifying.
        - Never guess or assume details about Suri Marketing’s services, pricing, or results.
        - Only use information provided in the conversation or system context.
        - If the user shows repeated interest or asks multiple detailed questions, move more directly toward suggesting a discovery call.
        - If the user hesitates (e.g. "not sure", "seems expensive"), acknowledge it briefly and respond calmly without pressure.
        - If a service, feature, or capability is asked about and is not offered, clearly say no in a natural way. Do not redirect unless the question is completely unrelated to Suri Marketing.
        
        ## Voice
        - Use contractions. "We're," "you're," "don't," "can't." Always.
        - Short sentences. Fragments are fine.
        - Skip marketing-speak: no "end-to-end solutions," "leverage," "holistic," etc.
        - Don't end every reply with a question. Only ask one when it genuinely moves the conversation forward.
        - Bullets only for 3+ parallel items. Default to prose.
        - If someone's rude, stay polite and steer back: "Here to help with Suri Marketing — let me know if there's something I can sort for you."

        ## Lead Qualification & Conversion Behaviour
        - If the user shows interest, ask simple questions to understand:
        - Their business
        - Their goals
        - Their current situation

        - Use this to:
        - Personalise responses
        - Make the service feel relevant

        - When the user is:
        - Curious → educate
        - Engaged → qualify
        - Interested → guide to booking

        ---

        ### When to Suggest a Discovery Call

        Suggest a call when:
        - The user asks about pricing or services in detail
        - The user asks “will this work for me?”
        - The user shows clear interest
        - The conversation reaches a natural pause

        ---

        ### How to Suggest It

        Keep it low-pressure and natural:

        Examples:
        - “Might be easier to walk through this on a quick call — want me to set that up?”
        - “We could go through this properly on a quick discovery call if you want”
        - “Happy to show you exactly how this would work for your business on a call”

        - You are representing a real company, so responses should feel grounded, confident, and aligned with a real service offering — not hypothetical
        """

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "supersecretkey")
app.config["UPLOAD_FOLDER"] = "static/files"
app.config["OCR_PROCESSING_FOLDER"] = "static/ocr"
app.config["EXTRACTED_TEXT_FOLDER"] = "static/extracted_text"
api = Api(app)

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=20)
sales_controller = SalesController(client=client)

response = client

# Initialise Postgres Database
db = Database("document_database.db", "documents")
db.init_schema()

# Retrieval alpha — read from RETRIEVAL_STRATEGY in .env (hybrid=0.5, dense=1.0, sparse=0.0)
retrieval_alpha = get_retrieval_alpha()


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


_LONDON = ZoneInfo("Europe/London")
_BUSINESS_START = 9   # hour
_BUSINESS_END = 17    # hour


def _build_day_map() -> str:
    """Plain-English lookup table injected into the controller prompt."""
    today = datetime.now(_LONDON).date()
    lines = []
    for i in range(1, 15):
        d = today + timedelta(days=i)
        lines.append(f"  - {d.strftime('%A').lower()}: {d.strftime('%-d %B %Y')}")
    return "\n".join(lines)


def _build_day_lookup() -> dict:
    """Map of lowercase weekday name → list of upcoming dates (14-day horizon)."""
    today = datetime.now(_LONDON).date()
    lookup: dict[str, list] = {}
    for i in range(1, 15):
        d = today + timedelta(days=i)
        name = d.strftime("%A").lower()
        lookup.setdefault(name, []).append(d)
    return lookup


def _resolve_day(day_name: str, day_lookup: dict):
    """Resolve a day name ('thursday', 'next monday', 'tomorrow') to a date."""
    if not day_name:
        return None
    day_name = day_name.lower().strip()
    if day_name == "tomorrow":
        return datetime.now(_LONDON).date() + timedelta(days=1)
    is_next = day_name.startswith("next ")
    base = day_name[5:] if is_next else day_name
    dates = day_lookup.get(base, [])
    if not dates:
        return None
    return dates[1] if (is_next and len(dates) > 1) else dates[0]


def _resolve_booking_window(controller_output: dict, day_lookup: dict, lead_profile: dict) -> tuple:
    """Build (start_dt, end_dt) from intent fields, falling back to lead profile then 7-day default."""
    now = datetime.now(_LONDON)

    req_day = controller_output.get("requested_day") or lead_profile.get("requested_day")
    time_from = controller_output.get("requested_time_from") or lead_profile.get("requested_time_from")
    time_to = controller_output.get("requested_time_to") or lead_profile.get("requested_time_to")

    if req_day:
        resolved = _resolve_day(req_day, day_lookup)
        if resolved:
            try:
                if time_from:
                    h, m = map(int, time_from.split(":"))
                else:
                    h, m = _BUSINESS_START, 0
                start_dt = datetime(resolved.year, resolved.month, resolved.day, h, m, tzinfo=_LONDON)

                if time_to:
                    h2, m2 = map(int, time_to.split(":"))
                else:
                    h2, m2 = _BUSINESS_END, 0
                end_dt = datetime(resolved.year, resolved.month, resolved.day, h2, m2, tzinfo=_LONDON)

                if end_dt > now:
                    return start_dt, end_dt
            except (ValueError, AttributeError):
                pass

    return now, now + timedelta(days=7)


def _resolve_selected_slot(controller_output: dict, day_lookup: dict):
    """Resolve selected_slot_day + selected_slot_time → ISO string, or None."""
    day_name = controller_output.get("selected_slot_day")
    time_str = controller_output.get("selected_slot_time")
    if not day_name or not time_str:
        return None
    resolved = _resolve_day(day_name, day_lookup)
    if not resolved:
        return None
    try:
        h, m = map(int, time_str.split(":"))
        slot_dt = datetime(resolved.year, resolved.month, resolved.day, h, m, tzinfo=_LONDON)
        return slot_dt.isoformat()
    except (ValueError, AttributeError):
        return None


def _handle_offer_slots(controller_output: dict, day_lookup: dict, lead_profile: dict) -> dict:
    try:
        start, end = _resolve_booking_window(controller_output, day_lookup, lead_profile)
        print(f"[SLOTS] Searching {start.isoformat()} → {end.isoformat()}")
        slots = get_available_slots(start, end)
        print(f"[SLOTS] Found: {slots}")
        if not slots:
            return {"status": "no_slots_available"}
        return {"status": "slots_available", "slots": slots}
    except cal_ex.CalendarAuthError:
        return {"status": "calendar_unavailable"}
    except cal_ex.CalendarServiceError as e:
        print(f"[BOOKING] Calendar service error: {e}")
        return {"status": "calendar_error"}


def _handle_book_appointment(controller_output: dict, day_lookup: dict, lead_profile: dict, session_id: str) -> dict:
    slot_iso = _resolve_selected_slot(controller_output, day_lookup)
    if not slot_iso:
        return {"status": "missing_slot"}

    try:
        slot_dt = datetime.fromisoformat(slot_iso)
        if slot_dt <= datetime.now(_LONDON):
            return {"status": "slot_in_past"}
    except ValueError:
        return {"status": "missing_slot"}

    name = lead_profile.get("contact_name") or "Guest"
    email = lead_profile.get("contact_email") or ""

    try:
        event_id = create_booking(slot_iso, name, email)
        db.save_booking(session_id, event_id, slot_iso, name, email)
        lead_profile["requested_day"] = None
        lead_profile["requested_time_from"] = None
        lead_profile["requested_time_to"] = None
        return {"status": "booked", "event_id": event_id, "slot_iso": slot_iso}
    except cal_ex.SlotAlreadyTakenError:
        london = ZoneInfo("Europe/London")
        now = datetime.now(london)
        try:
            slots = get_available_slots(now, now + timedelta(days=7))
        except cal_ex.CalendarServiceError:
            slots = []
        return {"status": "slot_taken", "slots": slots}
    except cal_ex.CalendarAuthError:
        return {"status": "calendar_unavailable"}
    except cal_ex.CalendarServiceError as e:
        print(f"[BOOKING] Error creating booking: {e}")
        return {"status": "calendar_error"}


def _handle_reschedule(controller_output: dict, day_lookup: dict, lead_profile: dict) -> dict:
    provided = controller_output.get("provided_email")

    if not provided:
        # No email yet — show slots so the user can pick a time while we ask for their email
        start, end = _resolve_booking_window(controller_output, day_lookup, lead_profile)
        try:
            slots = get_available_slots(start, end)
        except cal_ex.CalendarServiceError:
            slots = []
        return {"status": "needs_verification", "slots": slots}

    # Look up booking by email — works across sessions
    existing = db.get_booking_by_email(provided)
    if not existing:
        return {"status": "verification_failed"}

    new_slot_iso = _resolve_selected_slot(controller_output, day_lookup)
    if not new_slot_iso:
        return {"status": "missing_slot"}

    try:
        reschedule_booking(existing["event_id"], new_slot_iso)
        db.update_booking_slot(existing["booking_id"], new_slot_iso)
        return {"status": "rescheduled", "new_slot_iso": new_slot_iso}
    except cal_ex.EventNotFoundError:
        return {"status": "no_existing_booking"}
    except cal_ex.SlotAlreadyTakenError:
        london = ZoneInfo("Europe/London")
        now = datetime.now(london)
        try:
            slots = get_available_slots(now, now + timedelta(days=7))
        except cal_ex.CalendarServiceError:
            slots = []
        return {"status": "slot_taken", "slots": slots}
    except cal_ex.CalendarAuthError:
        return {"status": "calendar_unavailable"}
    except cal_ex.CalendarServiceError as e:
        print(f"[BOOKING] Reschedule error: {e}")
        return {"status": "calendar_error"}


# REST API for backend
chatbot_args = reqparse.RequestParser()
chatbot_args.add_argument("message", type=str, help="Message for LLM", required=True)
chatbot_args.add_argument("session_id", type=str, required=False)


# API that handles communication with Open AI
class ChatbotAPI(Resource):
    def post(self):
        args = chatbot_args.parse_args()
        user_message = args["message"]
        session_id = args["session_id"]

        # Retrieve Information
        history = db.get_conversation_history(session_id, limit=10)
        lead_profile = db.get_lead_profile(session_id)
        last_actions = db.get_last_actions(session_id, limit=3)

        sales_turn = db.get_sales_turn_count(session_id)
        print(f"[DEBUG] session_id={session_id!r}  sales_turn={sales_turn!r}")

        # Invoke Controller's output
        now_london = datetime.now(_LONDON)
        current_datetime_str = now_london.isoformat()
        current_date_label = now_london.strftime("%A %-d %B %Y")
        day_map = _build_day_map()
        day_lookup = _build_day_lookup()
        controller_output = sales_controller.run_controller(
            user_message=user_message,
            history=history,
            lead_profile=lead_profile,
            last_actions=last_actions,
            sales_turn=sales_turn,
            current_datetime=current_datetime_str,
            current_date_label=current_date_label,
            day_map=day_map
        )

        # apply guardrails to controller output
        controller_output = sales_controller.apply_guardrails(
            controller_output=controller_output,
            lead_profile=lead_profile,
            sales_turn=sales_turn
        )

        # extract parameter from controller output
        next_action = controller_output["next_action"]
        controller_reason = controller_output.get("reason", "")
        requires_rag = controller_output.get("requires_rag", False)
        email_rejected = controller_output.get("email_rejected", False)

        # Update lead profile
        lead_profile = sales_controller.merge_profile(
            lead_profile,
            controller_output.get("lead_profile_updates", {})
        )

        # Persist day/time preference so follow-up slot requests remember the window
        for field in ("requested_day", "requested_time_from", "requested_time_to"):
            val = controller_output.get(field)
            if val is not None:
                lead_profile[field] = val

        rag_context = ""

        if controller_output.get("requires_rag"):
            rag_context = retrieve_relevant_chunks(client, user_message, retrieval_alpha)

        booking_result = None

        if controller_output.get("requires_booking_tool"):
            action = next_action
            if action == "OFFER_AVAILABLE_SLOTS":
                booking_result = _handle_offer_slots(controller_output, day_lookup, lead_profile)
            elif action == "BOOK_APPOINTMENT":
                booking_result = _handle_book_appointment(controller_output, day_lookup, lead_profile, session_id)
            elif action == "RESCHEDULE_APPOINTMENT":
                booking_result = _handle_reschedule(controller_output, day_lookup, lead_profile)
            elif action == "CONFIRM_BOOKING":
                booking_result = {"status": "confirmed"}

        # 6. Generate final customer-facing reply
        assistant_reply = generate_final_response(
            client=client,
            user_message=user_message,
            history=history,
            lead_profile=lead_profile,
            next_action=next_action,
            controller_reason=controller_reason,
            requires_rag=requires_rag,
            rag_context=rag_context,
            booking_result=booking_result,
            email_rejected=email_rejected
        )

        # 7. Save everything
        db.save_message(session_id, "user", user_message)
        db.save_message(session_id, "assistant", assistant_reply, used_action=next_action)
        db.save_lead_profile(session_id, lead_profile)

        return {"response": assistant_reply}, 200


api.add_resource(ChatbotAPI, "/api")


# REST API to poll document status (status, has_text)
class DocumentStatusAPI(Resource):
    @login_required
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
    @login_required
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
@login_required
def get_documents():
    try:
        rows = db.get_all_documents()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    documents = [
        {
            "doc_id": row[0],
            "source_type": row[1],
            "file_name": row[2],
            "file_size": row[3],
            "file_type": row[4],
            "uploaded_date": str(row[5]),
            "status": row[8],
            "in_kb": row[12],
        }
        for row in rows
    ]

    return jsonify({"documents": documents}), 200


# REST API to mark a document as added to the knowledge base
@app.route("/document/<string:doc_id>/add-to-kb", methods=["POST"])
@login_required
def add_to_kb(doc_id):
    doc_type = db.get_doc_type(doc_id)
    db.set_kb_status(doc_id, "processing")
    threading.Thread(target=_chunk_and_embed_job, args=(doc_id,)).start()
    return jsonify({"status": "processing"}), 200


@app.route("/document/<string:doc_id>/kb-status", methods=["GET"])
@login_required
def get_kb_status(doc_id):
    kb_status = db.get_kb_status(doc_id)
    return jsonify({"kb_status": kb_status}), 200


def _chunk_and_embed_job(doc_id):
    strategy = get_chunking_strategy()
    print(
        f"\n[KB] Starting chunk + embed job for doc: {doc_id} (strategy: {strategy.value})"
    )
    try:
        # fetch extracted text from S3
        s3_bucket, s3_key = db.get_extracted_text_file_path(doc_id)
        text = _get_file_text(s3, s3_bucket, s3_key)

        # chunk using selected strategy
        print(f"[KB] Chunking text ({len(text)} chars)...")
        if strategy == ChunkingStrategy.SECTION_AND_SEMANTIC:
            chunks = chunk_sections_with_fallback(
                text, api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            clean_text = re.sub(r"\[HEADING\]\s*", "", text)
            if strategy == ChunkingStrategy.RECURSIVE_TOKEN:
                chunks = chunk_text_recursive(clean_text)
            elif strategy == ChunkingStrategy.FIXED_SIZE:
                chunks = chunk_text_fixed(clean_text)
            else:  # SEMANTIC_ONLY
                chunks = chunk_text(clean_text, api_key=os.getenv("OPENAI_API_KEY"))
        print(f"[KB] Produced {len(chunks)} chunks.")

        # embed (dense)
        embedded_chunks = embed_chunks(client, chunks)

        # drop chunks with no alphabetic content — BM25 cannot encode them
        before = len(embedded_chunks)
        embedded_chunks = [c for c in embedded_chunks if any(ch.isalpha() for ch in c["text"])]
        if len(embedded_chunks) < before:
            print(f"[KB] Filtered out {before - len(embedded_chunks)} punctuation-only chunk(s).")

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
            if any(not sv.get("indices") for sv in sparse_vectors):
                print("[KB] Empty sparse vectors detected (stale encoder) — refitting on current chunks.")
                encoder = bm25_enc.fit_and_save(texts, s3, bucket_name)
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


def _crawl_and_embed_job(url_id, url):
    strategy = get_chunking_strategy()
    print(
        f"\n[WEB] Starting crawl + embed job for url_id: {url_id} (strategy: {strategy.value})"
    )
    try:
        db.set_url_crawl_status(url_id, "processing")

        # crawl URL and get markdown
        raw_markdown = asyncio.run(crawl_url(url))
        print(f"[WEB] Crawled {url} — {len(raw_markdown)} chars of markdown")
        print(
            f"\n[WEB] ── Full extracted markdown ──────────────────────\n{raw_markdown}\n──────────────────────────────────────────────────────\n"
        )
        # clean noise (all strategies)
        cleaned = clean_markdown(raw_markdown)
        print(
            f"\n[WEB] ── Full cleaned markdown ──────────────────────\n{cleaned}\n──────────────────────────────────────────────────────\n"
        )
        # clean noise (all strategies)

        # section-based only: detect dominant header and insert [HEADING] markers
        if strategy == ChunkingStrategy.SECTION_AND_SEMANTIC:
            dominant = detect_dominant_level(cleaned)
            print(f"[WEB] Dominant header tag selected: {dominant}")
            text = markdown_to_sections(cleaned)
            print(
                f"\n[WEB] ── Section Chunked Markdown ──────────────────────\n{text}\n──────────────────────────────────────────────────────\n"
            )

        else:
            text = cleaned

        # upload processed text to S3
        s3_key = f"{url_id}.txt"
        s3.meta.client.put_object(
            Bucket=bucket_name, Key=s3_key, Body=text.encode("utf-8")
        )
        db.set_url_text_path(url_id, bucket_name, s3_key)

        # chunk using selected strategy
        print(f"[WEB] Chunking text ({len(text)} chars)...")
        if strategy == ChunkingStrategy.SECTION_AND_SEMANTIC:
            chunks = chunk_sections_with_fallback(
                text, api_key=os.getenv("OPENAI_API_KEY")
            )
        elif strategy == ChunkingStrategy.RECURSIVE_TOKEN:
            chunks = chunk_text_recursive(text)
        elif strategy == ChunkingStrategy.FIXED_SIZE:
            chunks = chunk_text_fixed(text)
        else:  # SEMANTIC_ONLY
            chunks = chunk_text(text, api_key=os.getenv("OPENAI_API_KEY"))
        print(f"[WEB] Produced {len(chunks)} chunks.")
        for i, chunk in enumerate(chunks):
            print(
                f"\n[WEB] ── Chunk {i+1}/{len(chunks)} ──────────────────────────────\n{chunk}\n"
            )

        # embed (dense)
        embedded_chunks = embed_chunks(client, chunks)

        # drop chunks with no alphabetic content — BM25 cannot encode them
        before = len(embedded_chunks)
        embedded_chunks = [c for c in embedded_chunks if any(ch.isalpha() for ch in c["text"])]
        if len(embedded_chunks) < before:
            print(f"[WEB] Filtered out {before - len(embedded_chunks)} punctuation-only chunk(s).")

        # build chunk rows for webpage_chunks
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
            if any(not sv.get("indices") for sv in sparse_vectors):
                print("[WEB] Empty sparse vectors detected (stale encoder) — refitting on current chunks.")
                encoder = bm25_enc.fit_and_save(texts, s3, bucket_name)
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
        print(f"[WEB] Upserting {len(chunks_for_upsert)} hybrid vectors to Pinecone...")
        pinecone_index = get_or_create_index(
            api_key=os.getenv("PINECONE_API_KEY"),
            index_name=os.getenv("PINECONE_INDEX_NAME"),
        )
        upsert_chunks(pinecone_index, url_id, chunks_for_upsert, sparse_vectors)

        # persist chunk rows to webpage_chunks table
        db.insert_webpage_chunks(url_id, chunk_rows)
        print(f"[WEB] Inserted {len(chunk_rows)} rows into webpage_chunks.")

        # refit and save BM25 encoder with updated corpus (docs + webpages)
        all_chunks = db.get_all_chunks()
        bm25_enc.fit_and_save(
            [c["text"] for c in all_chunks],
            s3,
            bucket_name,
        )

        # mark as in KB
        db.set_url_in_kb(url_id)
        db.set_url_kb_status(url_id, "success")
        db.set_url_crawl_status(url_id, "success")
        print(f"[WEB] URL {url_id} marked as in_kb=TRUE.")

    except Exception as e:
        print(f"[WEB] Error during crawl + embed job: {e}")
        db.set_url_crawl_status(url_id, "failed")
        db.set_url_kb_status(url_id, "failed")
        db.set_url_error(url_id, str(e))


@app.route("/query", methods=["POST"])
@login_required
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
@login_required
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

    return (
        jsonify(
            {
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
            }
        ),
        200,
    )


@app.route("/config/alpha", methods=["POST"])
@login_required
def set_alpha():
    global retrieval_alpha
    data = request.get_json()
    alpha = data.get("alpha")
    if (
        alpha is None
        or not isinstance(alpha, (int, float))
        or not (0.0 <= alpha <= 1.0)
    ):
        return jsonify({"error": "alpha must be a number between 0.0 and 1.0"}), 400
    retrieval_alpha = float(alpha)
    print(f"[config] retrieval_alpha updated to {retrieval_alpha}")
    return jsonify({"alpha": retrieval_alpha}), 200


# Admin Dashboard
class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload File")


@app.route("/admin", methods=["GET", "POST"])
@login_required
def admin():
    form = UploadFileForm()

    if request.method == "GET":
        return render_template("admin.html", form=form, active_page="files")

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
            upload_date=datetime.now(),
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
@login_required
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


@app.route("/sitemap", methods=["GET"])
@login_required
def sitemap():
    return render_template("sitemap.html", active_page="sitemap")


@app.route("/parse", methods=["POST"])
@login_required
def parse_urls():
    data = request.get_json()
    urls = data.get("urls", [])

    if not urls:
        return jsonify({"error": "No URLs provided"}), 400

    queued = []
    for url in urls:
        domain = urlparse(url).netloc
        url_id = db.create_url(url, domain)
        t = threading.Thread(
            target=_crawl_and_embed_job, args=(url_id, url), daemon=True
        )
        t.start()
        queued.append({"url": url, "url_id": url_id})

    return jsonify({"queued": queued}), 202


@app.route("/webpage/<url_id>/status", methods=["GET"])
@login_required
def webpage_status(url_id):
    status = db.get_webpage_status(url_id)
    return jsonify(status)

@app.route("/analytics")
@login_required
def analytics():
    return render_template("analytics.html", active_page="analytics")

@app.route("/analytics/summary")
@login_required
def analytics_summary():
    days = request.args.get("days", 7, type=int)
    days = days if days in (7, 30, 90) else 7
    data = db.get_analytics_summary(days)
    return jsonify(data)

@app.route("/analytics/bookings")
@login_required
def analytics_bookings():
    days = request.args.get("days", 7, type=int)
    days = days if days in (7, 30, 90) else 7
    bookings = db.get_bookings_for_period(days)
    return jsonify(bookings)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    data = request.get_json()
    username = (data or {}).get("username", "")
    password = (data or {}).get("password", "")
    stored_hash = os.getenv("ADMIN_PASSWORD_HASH", "")
    if (username == os.getenv("ADMIN_USERNAME", "") and
            stored_hash and
            bcrypt.checkpw(password.encode(), stored_hash.encode())):
        session['logged_in'] = True
        return jsonify({"ok": True})
    return jsonify({"message": "Invalid username or password."}), 401

@app.route("/logout")
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

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
