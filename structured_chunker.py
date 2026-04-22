from chunking_evaluation.utils import openai_token_count
from chunker import chunk_text

HEADING_PREFIX = "[HEADING]"
MIN_SECTION_WORDS = 10
MERGE_WARNING_THRESHOLD = 0.5
MAX_SECTION_TOKENS = 400


def split_into_sections(text: str) -> list[dict]:
    """
    Split heading-marked text into flat sections.

    Each line beginning with '[HEADING]' starts a new section.
    Any content before the first heading becomes chunk_id=0 with heading="".

    Short-section guardrail:
      - Sections with body text < MIN_SECTION_WORDS words are merged into the
        next section if one exists; otherwise left as-is.
      - If >= 50% of sections required merging, a warning is logged.
    """
    lines = text.splitlines()

    # Split into raw sections at [HEADING] boundaries
    raw_sections = []
    current_heading = ""
    current_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(HEADING_PREFIX):
            # Save the previous section
            raw_sections.append({
                "heading": current_heading,
                "body": "\n".join(current_lines).strip(),
            })
            current_heading = stripped[len(HEADING_PREFIX):].strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Append final section
    raw_sections.append({
        "heading": current_heading,
        "body": "\n".join(current_lines).strip(),
    })

    # Remove leading empty preamble if the doc starts with a heading immediately
    if raw_sections and raw_sections[0]["heading"] == "" and raw_sections[0]["body"] == "":
        raw_sections.pop(0)

    if not raw_sections:
        return []

    # Apply short-section guardrail — merge forward if body < MIN_PARENT_WORDS
    merge_count = 0
    merged = []
    i = 0
    while i < len(raw_sections):
        section = raw_sections[i]
        word_count = len(section["body"].split())

        if word_count < MIN_SECTION_WORDS and i + 1 < len(raw_sections):
            # Merge into next section
            next_section = raw_sections[i + 1]
            combined_body = (section["body"] + "\n\n" + next_section["body"]).strip()
            raw_sections[i + 1] = {
                "heading": next_section["heading"] if next_section["heading"] else section["heading"],
                "body": combined_body,
            }
            merge_count += 1
            i += 1  # skip current, next will be processed on next iteration
        else:
            merged.append(section)
            i += 1

    total = len(raw_sections)
    if total > 0 and merge_count / total >= MERGE_WARNING_THRESHOLD:
        print(
            f"[structured_chunker] WARNING: {merge_count}/{total} sections were merged due to "
            f"short body text (<{MIN_PARENT_WORDS} words). Heading detection may be unreliable."
        )

    # Assign chunk_ids
    sections = []
    for idx, section in enumerate(merged):
        sections.append({
            "chunk_id": idx,
            "heading": section["heading"],
            "text": (
                f"{section['heading']}\n\n{section['body']}"
                if section["heading"]
                else section["body"]
            ).strip(),
        })

    return sections


def chunk_sections_with_fallback(text: str, api_key: str) -> list[str]:
    """
    Adaptive chunking for the section_and_semantic strategy:
      1. Split at [HEADING] markers into sections.
      2. If no sections found, fall back to semantic chunking on the full text.
      3. For each section, if it exceeds MAX_SECTION_TOKENS, break it down
         further with semantic chunking; otherwise keep it as-is.
    Returns a flat list[str] of final chunks.
    """
    sections = split_into_sections(text)

    if not sections:
        print("[structured_chunker] No headings found — falling back to semantic chunking on full text.")
        return chunk_text(text, api_key=api_key)

    final_chunks = []
    for section in sections:
        token_count = openai_token_count(section["text"])
        if token_count > MAX_SECTION_TOKENS:
            print(
                f"[structured_chunker] Section '{section['heading'] or '(no heading)'}' "
                f"is {token_count} tokens (>{MAX_SECTION_TOKENS}) — applying semantic chunking."
            )
            sub_chunks = chunk_text(section["text"], api_key=api_key)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(section["text"])

    print(f"[structured_chunker] Produced {len(final_chunks)} final chunks from {len(sections)} sections.")
    return final_chunks
