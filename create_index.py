import pymupdf4llm
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# 1. Load the Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Extract with Page Metadata
print("Extracting book with page metadata...")
# This returns a list of dictionaries: [{"text": "...", "metadata": {"page": 0}}, ...]
pages = pymupdf4llm.to_markdown("databook.pdf", page_chunks=True)

chunks = []
metadata = []

for p in pages:
    page_text = p['text']
    page_num = p['metadata']['page'] + 1 # Convert 0-indexed to 1-indexed
    
    # Split the page into smaller paragraphs
    sub_chunks = [c.strip() for c in page_text.split("\n\n") if len(c.strip()) > 100]
    
    for sc in sub_chunks:
        chunks.append(sc)
        metadata.append({"page": page_num})

# 3. Create Embeddings & FAISS
print(f"Indexing {len(chunks)} chunks...")
embeddings = model.encode(chunks)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

# 4. Save Index and Metadata
faiss.write_index(index, "medical_db.index")
with open("chunks_with_meta.pkl", "wb") as f:
    pickle.dump({"chunks": chunks, "metadata": metadata}, f)

print("Ingestion Complete with Page Numbers!")