"""
Integration tests — require live PostgreSQL, OpenAI, and Pinecone.

Run with:
    pytest tests/test_integration.py -v -m integration
"""
import os
import pytest
import psycopg2 as pg2
from unittest.mock import patch

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Shared teardown helper
# ---------------------------------------------------------------------------

_DB_PARAMS = dict(
    host="localhost", dbname="postgres", user="postgres", password="9999", port=5432
)


def _raw_conn():
    return pg2.connect(**_DB_PARAMS)


# ===========================================================================
# Phase 1 — Text processing pipeline (no live services)
# ===========================================================================

from web_chunker import clean_markdown, markdown_to_sections
from structured_chunker import split_into_sections
from chunker import chunk_text_recursive


_REALISTIC_MARKDOWN = """\
## About Us
Suri Marketing is a social media marketing agency based in Birmingham with over five years \
of experience helping local businesses grow their online presence through targeted social \
media campaigns and paid advertising.

## Our Services
We offer a comprehensive range of digital marketing services including social media \
management, paid advertising on Meta and Google, content creation, SEO optimisation, \
and email marketing campaigns tailored to your business goals.

## Read More
Some junk CTA content that should be stripped from the final output entirely.

## Pricing
Our pricing is flexible and starts from five hundred pounds per month depending on the \
scope of work. We offer bespoke packages to suit businesses of all sizes from startups \
to established companies looking to grow.

## Get Started
Another call-to-action heading that the pipeline should filter out automatically.

## Contact
Reach us at hello@surimarketing.co.uk or call us on 0121 123 4567 to discuss your \
marketing needs with a member of the team.\
"""


def _run_pipeline(markdown: str) -> list[str]:
    cleaned = clean_markdown(markdown)
    marked = markdown_to_sections(cleaned)
    sections = split_into_sections(marked)
    chunks: list[str] = []
    for section in sections:
        text = section["text"]
        if section.get("heading"):
            text = f"{section['heading']}\n{text}"
        chunks.extend(chunk_text_recursive(text))
    return [c for c in chunks if c.strip()]


class TestTextProcessingPipeline:
    def test_full_pipeline_produces_non_empty_chunks(self):
        chunks = _run_pipeline(_REALISTIC_MARKDOWN)
        assert len(chunks) > 0
        assert all(c.strip() for c in chunks)

    def test_cta_phrases_absent_from_final_chunks(self):
        chunks = _run_pipeline(_REALISTIC_MARKDOWN)
        combined = "\n".join(chunks).lower()
        for cta in ("read more", "get started"):
            assert cta not in combined, f"CTA phrase '{cta}' survived the pipeline"

    def test_heading_content_survives_into_chunks(self):
        chunks = _run_pipeline(_REALISTIC_MARKDOWN)
        combined = "\n".join(chunks).lower()
        assert "services" in combined

    def test_pipeline_handles_markdown_with_no_headings(self):
        plain = (
            "Suri Marketing is a social media agency based in Birmingham. "
            "We help local businesses grow their online presence through targeted "
            "campaigns and strategic content creation that drives real measurable results "
            "for businesses of all sizes across the region."
        )
        chunks = _run_pipeline(plain)
        assert len(chunks) >= 1


# ===========================================================================
# Phase 2 — Database CRUD lifecycle (real PostgreSQL)
# ===========================================================================

from database import Database
from enums import DocumentStatus


@pytest.fixture(scope="module")
def db():
    instance = Database(database_path="localhost", doc_table_name="documents")
    instance.init_schema()
    return instance


@pytest.fixture
def new_doc(db):
    """Creates a document row, yields its doc_id, deletes it on teardown."""
    doc_id = db.create(
        source_type="upload",
        name="it-test-file.pdf",
        size="1024",
        type="PDF",
        upload_date="2026-01-01",
        s3_file_bucket="test-bucket",
        s3_file_key="it-test/file.pdf",
        s3_extracted_text_bucket="test-bucket",
        s3_extracted_text_key="it-test/extracted.txt",
    )
    yield doc_id
    conn = _raw_conn()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM document_chunks WHERE doc_id = %s", (doc_id,))
        cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
        conn.commit()
    finally:
        cur.close()
        conn.close()


_TEST_SESSION = "it-test-session-001"


@pytest.fixture
def clean_session():
    """Yields a test session_id and deletes all its rows on teardown."""
    yield _TEST_SESSION
    conn = _raw_conn()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM conversation_messages WHERE session_id = %s", (_TEST_SESSION,))
        cur.execute("DELETE FROM lead_profiles WHERE session_id = %s", (_TEST_SESSION,))
        conn.commit()
    finally:
        cur.close()
        conn.close()


class TestDatabaseCRUD:
    def test_create_document_status_is_created(self, new_doc, db):
        assert db.get_status(new_doc) == DocumentStatus.CREATED

    def test_valid_status_transitions(self, new_doc, db):
        db.transition_status(new_doc, DocumentStatus.PROCESSING)
        assert db.get_status(new_doc) == DocumentStatus.PROCESSING
        db.transition_status(new_doc, DocumentStatus.SUCCESS)
        assert db.get_status(new_doc) == DocumentStatus.SUCCESS

    def test_invalid_transition_raises(self, new_doc, db):
        # CREATED → SUCCESS skips PROCESSING — must raise
        with pytest.raises(Exception, match="Invalid"):
            db.transition_status(new_doc, DocumentStatus.SUCCESS)

    def test_insert_chunks_appears_in_get_all_chunks(self, new_doc, db):
        rows = [
            {"chunk_id": 0, "text": "Integration test chunk A", "heading": "Test"},
            {"chunk_id": 1, "text": "Integration test chunk B", "heading": "Test"},
        ]
        db.insert_chunks(new_doc, rows)
        all_chunks = db.get_all_chunks()
        doc_chunks = [c for c in all_chunks if c["doc_id"] == new_doc]
        assert len(doc_chunks) == 2
        texts = {c["text"] for c in doc_chunks}
        assert "Integration test chunk A" in texts
        assert "Integration test chunk B" in texts

    def test_save_and_retrieve_messages_in_order(self, db, clean_session):
        session_id = clean_session
        db.save_message(session_id, "user", "hello there")
        db.save_message(session_id, "assistant", "hey, how can I help?")
        db.save_message(session_id, "user", "what do you offer?")
        history = db.get_conversation_history(session_id)
        assert len(history) == 3
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "hello there"
        assert history[-1]["content"] == "what do you offer?"

    def test_sales_turn_count_only_counts_assistant(self, db, clean_session):
        session_id = clean_session
        db.save_message(session_id, "user", "hi")
        db.save_message(session_id, "assistant", "hey")
        db.save_message(session_id, "user", "cool")
        db.save_message(session_id, "assistant", "great")
        count = db.get_sales_turn_count(session_id)
        assert count == 2

    def test_save_and_retrieve_lead_profile(self, db, clean_session):
        session_id = clean_session
        profile = {
            "business_type": "restaurant",
            "location": "Birmingham",
            "main_goal": "get more bookings",
            "pain_confirmed": True,
            "booking_offered": False,
            "booking_completed": False,
            "current_marketing": None,
            "main_problem": None,
            "solution_matched": False,
            "objection": None,
            "contact_email": None,
            "contact_phone": None,
        }
        db.save_lead_profile(session_id, profile)
        retrieved = db.get_lead_profile(session_id)
        assert retrieved["business_type"] == "restaurant"
        assert retrieved["location"] == "Birmingham"
        assert retrieved["pain_confirmed"] is True


# ===========================================================================
# Phase 3 — Hybrid retrieval pipeline (real OpenAI + real Pinecone)
# ===========================================================================

from hybrid_retriever import hybrid_retrieve
import bm25_encoder as bm25_enc_module


@pytest.fixture(scope="module")
def openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@pytest.fixture(scope="module")
def pinecone_index():
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    return pc.Index(index_name)


class TestHybridRetrieval:
    def test_query_returns_correct_result_shape(self, openai_client, pinecone_index):
        results = hybrid_retrieve(
            query="what social media services do you offer?",
            client=openai_client,
            pinecone_index=pinecone_index,
            top_k=2,
        )
        assert isinstance(results, list)
        for r in results:
            assert {"doc_id", "chunk_id", "text", "score"} <= set(r.keys())

    def test_scores_are_valid_floats(self, openai_client, pinecone_index):
        results = hybrid_retrieve(
            query="pricing packages",
            client=openai_client,
            pinecone_index=pinecone_index,
        )
        for r in results:
            assert isinstance(r["score"], (int, float))

    def test_dense_only_fallback_when_encoder_is_none(self, openai_client, pinecone_index):
        with patch.object(bm25_enc_module, "_encoder", None):
            results = hybrid_retrieve(
                query="social media management",
                client=openai_client,
                pinecone_index=pinecone_index,
            )
        assert isinstance(results, list)
        for r in results:
            assert "text" in r

    def test_alpha_1_dense_only_mode_returns_results(self, openai_client, pinecone_index):
        results = hybrid_retrieve(
            query="paid advertising services",
            client=openai_client,
            pinecone_index=pinecone_index,
            alpha=1.0,
        )
        assert isinstance(results, list)


# ===========================================================================
# Phase 4 — Multi-turn conversation flow (real OpenAI, in-memory state)
# ===========================================================================

from sales_controller import SalesController
from response_generator import generate_final_response


@pytest.fixture(scope="class")
def sc(openai_client):
    return SalesController(client=openai_client)


class TestMultiTurnConversation:
    """
    Simulates a 3-turn conversation using real LLM calls.
    History and lead_profile accumulate as class attributes across turns.
    Tests must run in definition order (pytest default within a class).
    """

    history: list = []
    lead_profile: dict = SalesController.create_new_profile()

    def test_turn1_opening_question(self, sc, openai_client):
        user_msg = "what services do you offer?"
        raw = sc.run_controller(
            user_message=user_msg,
            history=self.history,
            lead_profile=self.lead_profile,
            last_actions=[],
            sales_turn=0,
        )
        result = sc.apply_guardrails(raw, self.lead_profile, sales_turn=0)
        assert result["next_action"] in SalesController.VALID_ACTIONS

        response = generate_final_response(
            client=openai_client,
            user_message=user_msg,
            history=self.history,
            lead_profile=self.lead_profile,
            next_action=result["next_action"],
            controller_reason=result.get("reason", ""),
            requires_rag=result["requires_rag"],
        )
        assert isinstance(response, str) and len(response) > 0

        self.__class__.history = self.history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": response},
        ]
        self.__class__.lead_profile = SalesController(client=None).merge_profile(
            self.lead_profile, result.get("lead_profile_updates", {})
        )

    def test_turn2_lead_qualification(self, sc, openai_client):
        user_msg = "we're a restaurant in Birmingham"
        raw = sc.run_controller(
            user_message=user_msg,
            history=self.history,
            lead_profile=self.lead_profile,
            last_actions=[],
            sales_turn=1,
        )
        result = sc.apply_guardrails(raw, self.lead_profile, sales_turn=1)
        updates = result.get("lead_profile_updates", {})
        assert any(
            updates.get(k) for k in ("business_type", "location")
        ), f"Expected business_type or location in lead_profile_updates, got: {updates}"

        response = generate_final_response(
            client=openai_client,
            user_message=user_msg,
            history=self.history,
            lead_profile=self.lead_profile,
            next_action=result["next_action"],
            controller_reason=result.get("reason", ""),
        )
        assert isinstance(response, str) and len(response) > 0

        self.__class__.history = self.history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": response},
        ]
        self.__class__.lead_profile = SalesController(client=None).merge_profile(
            self.lead_profile, updates
        )

    def test_turn3_guardrail_forces_offer_booking(self, sc, openai_client):
        user_msg = "sounds interesting, tell me more"
        raw = sc.run_controller(
            user_message=user_msg,
            history=self.history,
            lead_profile=self.lead_profile,
            last_actions=[],
            sales_turn=3,
        )
        result = sc.apply_guardrails(raw, self.lead_profile, sales_turn=3, threshold=3)
        assert result["next_action"] == "OFFER_BOOKING"

        response = generate_final_response(
            client=openai_client,
            user_message=user_msg,
            history=self.history,
            lead_profile=self.lead_profile,
            next_action=result["next_action"],
            controller_reason=result.get("reason", ""),
        )
        assert isinstance(response, str) and len(response) > 0
