# import module
from pdf2image import convert_from_path
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Function to create a file with the Files API
def create_file(file_path):
  with open(file_path, "rb") as file_content:
    result = client.files.create(
        file=file_content,
        purpose="vision",
    )
    return result.id

# convert pdf to images
file_name = 'doc2_organic_proposal.pdf'
images = convert_from_path(file_name)
doc_dir_name = file_name.split(".")[0]
directory_name = f'document_images/{doc_dir_name}'

all_extracted_texts = []

# create directory to store the images of pdf pages
try:
    os.makedirs(directory_name, exist_ok=True)
except OSError as e:
    print("Error:", e)

for page_no in range(len(images)):
    # convert page to png
    file_path = os.path.join(directory_name, 'page'+ str(page_no) +'.png')
    images[page_no].save(file_path, 'PNG')

    file_id = create_file(file_path)

    # Use Open AI to extract information
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[{
        "role": "system",
        "content": "You are a document-to-text conversion engine."
        },
            
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": 
                """
                You are a document-to-text conversion engine for building a searchable knowledge base.

                Convert this page image into clean, structured, chunk-ready plain text.

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
                
                """},
                {
                    "type": "input_image",
                    "file_id": file_id,
                },
            ],
        }],
    )

    # add extracted text from the page
    all_extracted_texts.append(f"--- PAGE {page_no} ---\n{response.output_text}")

all_extracted_texts = "\n\n".join(all_extracted_texts)

with open("demofile.txt", "a") as f:
  f.write(all_extracted_texts)

print("DONE")


