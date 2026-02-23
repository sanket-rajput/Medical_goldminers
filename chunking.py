import pymupdf4llm
import pathlib
import json

# Path to your 4000-page Merck Manual
pdf_path = "/workspaces/Medical_goldminers/The_Merck.pdf"
output_md = "book_content.md"
output_json = "book_structure.json"

print("Starting conversion of 4000+ pages. This will take a few minutes...")

# Step 1: Convert with Page Metadata
# Using page_chunks=True is vital for citing page numbers later
pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)

# Step 2: Save as a clean Markdown file for human reading
full_text = ""
structured_data = []

for page in pages:
    page_text = page['text']
    page_num = page['metadata']['page'] + 1  # Normalize to 1-based indexing
    
    # Add a hidden markup so we can identify page boundaries later
    page_header = f"\n\n\n"
    full_text += page_header + page_text
    
    # Store structured data for the next FAISS step
    structured_data.append({
        "page": page_num,
        "content": page_text
    })

# Save the Markdown version
pathlib.Path(output_md).write_bytes(full_text.encode("utf-8"))

# Save a JSON version (This makes Step 2: Indexing much faster)
with open(output_json, "w") as f:
    json.dump(structured_data, f)

print(f"Step 1 Complete!")
print(f"- Human-readable Markdown saved to: {output_md}")
print(f"- Structured JSON for FAISS saved to: {output_json}")