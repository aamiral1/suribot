import re
from collections import Counter


CTA_PHRASES = {
    "read more", "learn more", "click here", "click now", "find out more",
    "get started", "contact us", "buy now", "shop now", "sign up",
    "subscribe", "download", "view more", "see more", "show more",
    "explore", "discover", "get in touch", "book now", "order now", "browse",
}

HTML_TAG_LINE = re.compile(r'^\s*<[^>]+>\s*$')


def clean_markdown(text: str) -> str:
    cleaned = []
    for line in text.splitlines():
        s = line.strip()
        if HTML_TAG_LINE.match(s):
            continue
        content = re.sub(r'^#{1,6}\s*', '', s).strip().lower()
        if content in CTA_PHRASES:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def detect_dominant_level(markdown: str) -> str:
    """Returns the dominant header tag string (e.g. 'H2', 'H3') or 'None' if no headers found."""
    header_pattern = re.compile(r'^(#{2,4})\s+', re.MULTILINE)
    matches = header_pattern.findall(markdown)
    if not matches:
        return "None"
    level_counts = Counter(len(h) for h in matches)
    dominant = level_counts.most_common(1)[0][0]
    return f"H{dominant}"


def markdown_to_sections(markdown: str) -> str:
    """
    Converts markdown headers to [HEADING] markers at the dominant header level.
    The dominant level is whichever of ##/###/#### appears most frequently.
    Headers at other levels have their # symbols stripped (become plain text).
    If no headers are found, returns markdown unchanged.
    """
    header_pattern = re.compile(r'^(#{2,4})\s+(.+)$', re.MULTILINE)
    matches = header_pattern.findall(markdown)

    if not matches:
        return markdown

    level_counts = Counter(len(hashes) for hashes, _ in matches)
    dominant_level = level_counts.most_common(1)[0][0]

    def replace_header(m):
        hashes = m.group(1)
        title = m.group(2).strip()
        if len(hashes) == dominant_level:
            return f'[HEADING] {title}'
        return title

    return header_pattern.sub(replace_header, markdown)
