import re


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
