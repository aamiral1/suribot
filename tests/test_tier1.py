import os
import pytest

# ---------------------------------------------------------------------------
# Phase 1 — web_chunker.py
# ---------------------------------------------------------------------------

from web_chunker import clean_markdown, detect_dominant_level, markdown_to_sections


class TestCleanMarkdown:
    def test_html_only_line_is_removed(self):
        result = clean_markdown("<div>\nHello\n</div>")
        assert "<div>" not in result
        assert "</div>" not in result
        assert "Hello" in result

    def test_cta_phrase_heading_is_removed(self):
        result = clean_markdown("## Read More\nSome content")
        assert "Read More" not in result
        assert "Some content" in result

    def test_cta_phrase_plain_line_is_removed(self):
        result = clean_markdown("learn more\nSome content")
        assert "learn more" not in result

    def test_normal_content_is_preserved(self):
        text = "This is a normal paragraph.\nAnother line."
        assert clean_markdown(text) == text

    def test_empty_string_returns_empty(self):
        assert clean_markdown("") == ""

    def test_mixed_content_strips_only_cta(self):
        text = "## Our Services\n## Click Here\nWe offer great services."
        result = clean_markdown(text)
        assert "Our Services" in result
        assert "Click Here" not in result
        assert "We offer great services." in result


class TestDetectDominantLevel:
    def test_no_headers_returns_none_string(self):
        assert detect_dominant_level("Just some text") == "None"

    def test_only_h2_headers(self):
        md = "## Section One\n## Section Two\n## Section Three"
        assert detect_dominant_level(md) == "H2"

    def test_h3_dominant_over_h2(self):
        md = "## Intro\n### Part A\n### Part B\n### Part C"
        assert detect_dominant_level(md) == "H3"

    def test_h4_dominant(self):
        md = "#### A\n#### B\n#### C\n## Top"
        assert detect_dominant_level(md) == "H4"

    def test_empty_string_returns_none_string(self):
        assert detect_dominant_level("") == "None"


class TestMarkdownToSections:
    def test_no_headers_returns_unchanged(self):
        text = "Just plain text\nno headers here"
        assert markdown_to_sections(text) == text

    def test_h2_dominant_becomes_heading_marker(self):
        md = "## Section One\nContent A\n## Section Two\nContent B"
        result = markdown_to_sections(md)
        assert "[HEADING] Section One" in result
        assert "[HEADING] Section Two" in result

    def test_non_dominant_header_stripped_to_plain_text(self):
        md = "## Main\nContent\n## Main Two\nContent\n### Sub\nSub content"
        result = markdown_to_sections(md)
        assert "[HEADING] Main" in result
        assert "Sub" in result
        assert "### Sub" not in result
        assert "[HEADING] Sub" not in result

    def test_all_same_level_all_become_headings(self):
        md = "### A\ntext\n### B\ntext\n### C\ntext"
        result = markdown_to_sections(md)
        assert result.count("[HEADING]") == 3

    def test_empty_string_returns_empty(self):
        assert markdown_to_sections("") == ""


# ---------------------------------------------------------------------------
# Phase 2 — structured_chunker.py (split_into_sections only)
# ---------------------------------------------------------------------------

from structured_chunker import split_into_sections


class TestSplitIntoSections:
    def test_empty_string_returns_empty_list(self):
        assert split_into_sections("") == []

    def test_no_headings_single_section_no_heading(self):
        text = "Just some body text with no headings at all."
        sections = split_into_sections(text)
        assert len(sections) == 1
        assert sections[0]["heading"] == ""
        assert "body text" in sections[0]["text"]

    def test_normal_multi_section_split(self):
        text = (
            "[HEADING] About Us\nWe are a digital marketing agency based in Birmingham with years of experience.\n"
            "[HEADING] Services\nWe offer SEO, social media management, and paid advertising campaigns."
        )
        sections = split_into_sections(text)
        assert len(sections) == 2
        assert sections[0]["heading"] == "About Us"
        assert sections[1]["heading"] == "Services"
        assert sections[0]["chunk_id"] == 0
        assert sections[1]["chunk_id"] == 1

    def test_preamble_before_first_heading_is_included(self):
        text = (
            "This is introductory text that appears before any heading in the document.\n"
            "[HEADING] Section One\nThis section has enough words to avoid the merge guardrail triggering."
        )
        sections = split_into_sections(text)
        assert len(sections) == 2
        assert sections[0]["heading"] == ""
        assert "introductory text" in sections[0]["text"]

    def test_doc_starting_with_heading_drops_empty_preamble(self):
        text = "[HEADING] First Section\nBody content here"
        sections = split_into_sections(text)
        assert len(sections) == 1
        assert sections[0]["heading"] == "First Section"

    def test_short_section_merged_into_next(self):
        # "Short." is only 1 word — below MIN_SECTION_WORDS (10)
        text = "[HEADING] Brief\nShort.\n[HEADING] Full\n" + "word " * 15
        sections = split_into_sections(text)
        # The short section gets merged into the next, so only 1 section remains
        assert len(sections) == 1
        assert "Short." in sections[0]["text"]

    def test_last_short_section_kept_as_is(self):
        # Full first section, then a short last section with no next to merge into
        text = "[HEADING] Main\n" + "word " * 15 + "\n[HEADING] Tail\nTiny."
        sections = split_into_sections(text)
        tail = next((s for s in sections if s["heading"] == "Tail"), None)
        assert tail is not None
        assert "Tiny." in tail["text"]

    def test_merge_warning_printed_when_threshold_exceeded(self, capsys):
        # 3 short sections out of 4 total = 75% merged — exceeds 50% threshold
        short = "word\n"
        long_body = "word " * 15 + "\n"
        text = (
            f"[HEADING] A\n{short}"
            f"[HEADING] B\n{short}"
            f"[HEADING] C\n{short}"
            f"[HEADING] D\n{long_body}"
        )
        split_into_sections(text)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out


# ---------------------------------------------------------------------------
# Phase 3 — enums.py + config.py
# ---------------------------------------------------------------------------

from enums import AllowedFileTypes
import custom_exceptions as ex


class TestAllowedFileTypesFromFilename:
    def test_pdf_extension(self):
        assert AllowedFileTypes.from_filename("document.pdf") == AllowedFileTypes.PDF

    def test_docx_extension(self):
        assert AllowedFileTypes.from_filename("report.docx") == AllowedFileTypes.DOCX

    def test_png_uppercase_filename(self):
        assert AllowedFileTypes.from_filename("IMAGE.PNG") == AllowedFileTypes.PNG

    def test_mixed_case_extension(self):
        assert AllowedFileTypes.from_filename("file.Txt") == AllowedFileTypes.TXT

    def test_unsupported_extension_raises(self):
        with pytest.raises(ex.InvalidFileType):
            AllowedFileTypes.from_filename("file.xyz")

    def test_empty_string_raises(self):
        with pytest.raises(ex.InvalidFileType):
            AllowedFileTypes.from_filename("")

    def test_no_dot_raises(self):
        with pytest.raises(ex.InvalidFileType):
            AllowedFileTypes.from_filename("nodotfile")


from config import get_chunking_strategy, get_retrieval_alpha, ChunkingStrategy


class TestGetChunkingStrategy:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("CHUNKING_STRATEGY", raising=False)
        assert get_chunking_strategy() == ChunkingStrategy.SECTION_AND_SEMANTIC

    def test_recursive_token(self, monkeypatch):
        monkeypatch.setenv("CHUNKING_STRATEGY", "recursive_token")
        assert get_chunking_strategy() == ChunkingStrategy.RECURSIVE_TOKEN

    def test_fixed_size(self, monkeypatch):
        monkeypatch.setenv("CHUNKING_STRATEGY", "fixed_size")
        assert get_chunking_strategy() == ChunkingStrategy.FIXED_SIZE

    def test_uppercase_value_normalised(self, monkeypatch):
        monkeypatch.setenv("CHUNKING_STRATEGY", "FIXED_SIZE")
        assert get_chunking_strategy() == ChunkingStrategy.FIXED_SIZE

    def test_invalid_value_raises(self, monkeypatch):
        monkeypatch.setenv("CHUNKING_STRATEGY", "garbage")
        with pytest.raises(ValueError, match="Invalid CHUNKING_STRATEGY"):
            get_chunking_strategy()


class TestGetRetrievalAlpha:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("RETRIEVAL_STRATEGY", raising=False)
        assert get_retrieval_alpha() == 0.5

    def test_hybrid_returns_half(self, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_STRATEGY", "hybrid")
        assert get_retrieval_alpha() == 0.5

    def test_dense_returns_one(self, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_STRATEGY", "dense")
        assert get_retrieval_alpha() == 1.0

    def test_sparse_returns_zero(self, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_STRATEGY", "sparse")
        assert get_retrieval_alpha() == 0.0

    def test_unknown_value_defaults_to_half(self, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_STRATEGY", "unknown_mode")
        assert get_retrieval_alpha() == 0.5


# ---------------------------------------------------------------------------
# Phase 4 — response_generator.py (_clean_response + book_appointment)
# ---------------------------------------------------------------------------

from response_generator import _clean_response, book_appointment


class TestCleanResponse:
    def test_em_dash_replaced(self):
        assert " -" in _clean_response("good — bad")
        assert "—" not in _clean_response("good — bad")

    def test_triple_newlines_collapsed(self):
        result = _clean_response("line one\n\n\n\nline two")
        assert "\n\n\n" not in result
        assert "line one" in result
        assert "line two" in result

    def test_certainly_stripped(self):
        result = _clean_response("Certainly! Here is the answer.")
        assert not result.startswith("Certainly!")
        assert "Here is the answer." in result

    def test_absolutely_stripped(self):
        result = _clean_response("Absolutely! We can help.")
        assert not result.startswith("Absolutely!")

    def test_of_course_stripped(self):
        result = _clean_response("Of course! Let me explain.")
        assert not result.startswith("Of course!")

    def test_great_question_stripped(self):
        result = _clean_response("Great question! Here's the answer.")
        assert not result.startswith("Great question!")

    def test_sure_stripped(self):
        result = _clean_response("Sure! Happy to help.")
        assert not result.startswith("Sure!")

    def test_happy_to_help_stripped(self):
        result = _clean_response("Happy to help! Let me explain.")
        assert not result.startswith("Happy to help!")

    def test_clean_text_passes_through(self):
        text = "yeah paid ads is a big part of what we do"
        assert _clean_response(text) == text

    def test_whitespace_trimmed(self):
        assert _clean_response("  hello  ") == "hello"

    def test_banned_phrase_mid_sentence_not_stripped(self):
        text = "I said certainly! it was fine"
        result = _clean_response(text)
        assert "certainly!" in result


class TestBookAppointment:
    def test_returns_slots_available_status(self):
        result = book_appointment({})
        assert result["status"] == "slots_available"

    def test_returns_three_slots(self):
        result = book_appointment({})
        assert len(result["slots"]) == 3

    def test_each_slot_has_date_and_time(self):
        for slot in book_appointment({})["slots"]:
            assert "date" in slot
            assert "time" in slot
