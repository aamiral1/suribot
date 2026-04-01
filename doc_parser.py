from pdf2image import convert_from_path
import os
import datetime
from custom_exceptions import ExtractionTimeOut
from docx import Document

OCR_PROMPT = """
You are a document-to-text conversion engine for building a searchable knowledge base.

Convert the provided document into clean, structured, chunk-ready plain text.

Requirements:

Process the document page by page.

For each page, output in the exact format:
--- PAGE {page_number} ---
{extracted_text}

STRICT RULES:
- Output ONLY the extracted/converted content text.
- Do NOT describe the image.
- Do NOT summarise.
- Do NOT explain.
- Do NOT add commentary.
- Do NOT guess or hallucinate missing information.
- Preserve wording and numbers exactly as written.
- If text is unreadable, omit it.
- Preserve capitalization of headings.

SYMBOL CONVERSIONS:
- Checkmark/tick → "✓"
- Cross → "✗"
- Arrow → "→"
- Currency, percentages, dates → preserve exactly

GENERAL STRUCTURE RULES:
1) Normal Paragraph Pages:
- Preserve paragraphs.
- Preserve bullet points using "-" prefix.
- Keep headings separated by blank lines.

2) Comparison Layouts (tables, pricing grids, feature matrices):
- Do NOT output as rows/columns.
- Rewrite into separate self-contained sections per item/plan/package.
- Each section must include all associated facts (prices, quantities, features, limits).
- Format:
  [ITEM NAME]
  - fact
  - fact
  - includes: ...
  - price: ...

3) Process / Flow / Funnel Diagrams:
- Detect directional or sequential relationships.
- Convert to explicit ordered sequence using "→".
- Include descriptions under each step.
- Format:
  PROCESS:
  Step 1: ...
  - description
  →
  Step 2: ...
  - description
  →
  Step 3: ...

4) Step-by-Step Instructions / Roadmaps:
- Preserve order.
- Use numbered format:
  STEP 1: ...
  - details
  STEP 2: ...
  - details

5) Metrics / Results / Case Studies:
- Keep numeric results clearly tied to their labels.
- Format:
  [METRIC TITLE]
  - value: ...
  - timeframe: ...
  - context: ...

6) Multi-Column Marketing Layouts:
- Ignore visual column structure.
- Merge into logical reading order.
- Preserve section headings.

7) FAQs:
- Format clearly as:
  Q: ...
  A: ...

8) Testimonials / Quotes:
- Preserve quoted text.
- Attribute if visible.

9) Contact Information:
- Keep structured as:
  CONTACT:
  - phone:
  - email:
  - website:

FINAL REQUIREMENTS:
- Each logical section must be self-contained.
- Separate sections with a blank line.
- Ensure output is readable as standalone text without needing the original layout.
"""


def __create_file(client, file_path):
    with open(file_path, "rb") as f:
        result = client.files.create(file=f, purpose="vision")
    return result.id

def extract_from_pdf(client, pdf_file_path, images_dir_name):
    start_time = datetime.datetime.now()
    max_time = 300

    images = convert_from_path(pdf_file_path)

    doc_name = os.path.splitext(os.path.basename(pdf_file_path))[0]
    directory_name = os.path.join(images_dir_name, doc_name)
    os.makedirs(directory_name, exist_ok=True)


    all_extracted_texts = []

    for page_no, img in enumerate(images, start=1):
        delta_time = datetime.datetime.now() - start_time

        if delta_time.total_seconds() > max_time:
            raise ExtractionTimeOut("Document extraction exceeded max time allowed.")

        file_path = os.path.join(directory_name, f"page{page_no}.png")
        img.save(file_path, "PNG")

        file_id = None
        try:
            file_id = __create_file(client, file_path)

            response = client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {
                        "role": "system",
                        "content": "You are a document-to-text conversion engine.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": OCR_PROMPT},
                            {"type": "input_image", "file_id": file_id},
                        ],
                    },
                ],
            )

            # If your SDK supports it, this is fine:
            text = getattr(response, "output_text", None)
            if text is None:
                # fallback if output_text isn't available
                text = str(response)

            all_extracted_texts.append(f"--- PAGE {page_no} ---\n{text}".strip())

        except Exception as e:
            all_extracted_texts.append(
                f"--- PAGE {page_no} ---\n[ERROR extracting page: {e}]"
            )
        finally:
            if file_id is not None:
                try:
                    client.files.delete(file_id)
                except Exception:
                    pass

    return "\n\n".join(all_extracted_texts)

def extract_from_image(client, file_path):
    file_id = None
    try:
        file_id = __create_file(client, file_path)

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": "You are a document-to-text conversion engine.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": OCR_PROMPT},
                        {"type": "input_image", "file_id": file_id},
                    ],
                },
            ],
        )

        text = getattr(response, "output_text", None)
        if text is None:
            text = str(response)
        return text.strip()

    finally:
        if file_id is not None:
            try:
                client.files.delete(file_id)
            except Exception:
                pass


def extract_from_text_file(client, file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": "You are a document-to-text conversion engine.",
            },
            {
                "role": "user",
                "content": OCR_PROMPT + "\n\n" + content,
            },
        ],
    )

    text = getattr(response, "output_text", None)
    if text is None:
        text = str(response)
    return text.strip()


def extract_from_md(client, file_path):
    return extract_from_text_file(client, file_path)


def extract_doc_info(client, file_path):
    ext = os.path.splitext(file_path)[1].lower()
    print("file type: ", ext)
    
    if ext == ".pdf":
        images_dir = os.path.join(os.path.dirname(file_path), "extracted_images")
        return extract_from_pdf(client, file_path, images_dir)
    elif ext in (".png", ".jpg", ".jpeg"):
        return extract_from_image(client, file_path)
    elif ext == ".txt":
        return extract_from_text_file(client, file_path)
    elif ext == ".md":
        return extract_from_md(client, file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")