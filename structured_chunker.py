HEADING_PREFIX = "[HEADING]"
MIN_PARENT_WORDS = 10
MERGE_WARNING_THRESHOLD = 0.5


def split_into_parents(text: str) -> list[dict]:
    """
    Split heading-marked text into parent chunks.

    Each line beginning with '[HEADING]' starts a new parent section.
    Any content before the first heading becomes chunk_id=0 with heading="".

    Short-section guardrail:
      - Sections with body text < MIN_PARENT_WORDS words are merged into the
        next parent if one exists; otherwise left as-is.
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

        if word_count < MIN_PARENT_WORDS and i + 1 < len(raw_sections):
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
    parents = []
    for idx, section in enumerate(merged):
        parents.append({
            "chunk_id": idx,
            "heading": section["heading"],
            "text": (
                f"[HEADING] {section['heading']}\n\n{section['body']}"
                if section["heading"]
                else section["body"]
            ).strip(),
        })

    return parents
