#!/usr/bin/env python3
"""
Step 2: Generate sentence-transformer embeddings for all medicines.
Run after 1_excel_to_json.py:
    python3 tools/medicine_pipeline/2_generate_embeddings.py

Output:
    tools/medicine_pipeline/output/medicine_embeddings.npy   (float32 array)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

IN_JSON      = Path("tools/medicine_pipeline/output/medicines.json")
OUT_EMB      = Path("tools/medicine_pipeline/output/medicine_embeddings.npy")
BATCH_SIZE   = 512
MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    # Lazy import — only needed here
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed.")
        print("Run: pip3 install sentence-transformers")
        sys.exit(1)

    print(f"[1/4] Loading medicines from {IN_JSON} …")
    with open(IN_JSON, encoding="utf-8") as f:
        records = json.load(f)
    texts = [r["search_text"] for r in records]
    print(f"      {len(texts):,} search texts loaded.")

    print(f"[2/4] Loading model: {MODEL_NAME} …")
    model = SentenceTransformer(MODEL_NAME)

    print(f"[3/4] Encoding in batches of {BATCH_SIZE} …")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity via dot product
    ).astype("float32")
    elapsed = time.time() - t0
    print(f"      Done in {elapsed:.1f}s. Shape: {embeddings.shape}")

    print(f"[4/4] Saving embeddings to {OUT_EMB} …")
    OUT_EMB.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_EMB, embeddings)
    print(f"      Saved. File size: {OUT_EMB.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
