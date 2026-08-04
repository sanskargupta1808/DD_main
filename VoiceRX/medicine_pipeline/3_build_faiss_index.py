#!/usr/bin/env python3
"""
Step 3: Build FAISS HNSW index from pre-generated embeddings.
Run after 2_generate_embeddings.py:
    python3 tools/medicine_pipeline/3_build_faiss_index.py

Output:
    tools/medicine_pipeline/output/medicine.index
"""
import sys
import time
from pathlib import Path

import numpy as np

IN_EMB   = Path("tools/medicine_pipeline/output/medicine_embeddings.npy")
OUT_IDX  = Path("tools/medicine_pipeline/output/medicine.index")


def main() -> None:
    try:
        import faiss
    except ImportError:
        print("ERROR: faiss-cpu not installed.")
        print("Run: pip3 install faiss-cpu")
        sys.exit(1)

    print(f"[1/4] Loading embeddings from {IN_EMB} …")
    embeddings = np.load(IN_EMB).astype("float32")
    n, dim = embeddings.shape
    print(f"      Loaded {n:,} vectors of dimension {dim}.")

    print(f"[2/4] Building IndexHNSWFlat (M=32) …")
    t0 = time.time()
    # HNSW: fast approximate nearest-neighbour, no GPU needed, persists well
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200   # higher = better recall during build
    index.hnsw.efSearch = 64          # higher = better recall during query

    index.add(embeddings)
    elapsed = time.time() - t0
    print(f"      Index built in {elapsed:.1f}s. Total vectors: {index.ntotal:,}")

    print(f"[3/4] Saving index to {OUT_IDX} …")
    OUT_IDX.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(OUT_IDX))
    size_mb = OUT_IDX.stat().st_size / 1e6
    print(f"      Saved. File size: {size_mb:.1f} MB")

    # Quick sanity check
    print(f"[4/4] Sanity check — searching for 'paracetamol 650 tablet' …")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q_vec = model.encode(["paracetamol 650 tablet fever pain"],
                         normalize_embeddings=True).astype("float32")
    distances, indices = index.search(q_vec, 5)
    print(f"      Top-5 indices: {indices[0].tolist()}")
    print("      Done!")


if __name__ == "__main__":
    main()
