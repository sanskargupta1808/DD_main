#!/usr/bin/env python3
"""
Medicine Search FastAPI Server
Provides a /search endpoint used by the Flutter app.

Start with:
    uvicorn tools.medicine_pipeline.server:app --host 0.0.0.0 --port 8000
  OR from the repo root:
    python3 -m uvicorn tools.medicine_pipeline.server:app --host 0.0.0.0 --port 8000
"""
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import numpy as np
import faiss
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Paths (relative to repo root, or override via env vars) ──────────────────
BASE = Path(os.getenv("PIPELINE_DIR", str(Path(__file__).resolve().parent / "output")))

INDEX_PATH = BASE / "medicine.index"
META_PATH  = BASE / "medicine_meta.json"
TREE_PATH  = BASE / "bktree.pkl"
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "85.0"))
FAISS_TOP_K          = int(os.getenv("FAISS_TOP_K", "20"))
BKTREE_MAX_DIST      = int(os.getenv("BKTREE_MAX_DIST", "3"))


# ── Normaliser ────────────────────────────────────────────────────────────────
_FILLER = re.compile(
    r'\b(tablet|capsule|cap|tab|syrup|injection|inj|drop|gel|cream|ointment'
    r'|once|twice|thrice|daily|bd|tds|od|sos|mg|ml|g)\b',
    re.IGNORECASE
)

_NUMBER_WORDS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
    'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
    'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
    'hundred': 100, 'thousand': 1000
}

def words_to_numbers(text: str) -> str:
    """Converts spoken numbers like 'six fifty' to '650'."""
    words = text.split()
    out = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in _NUMBER_WORDS:
            val = 0
            current_group = 0
            while i < len(words) and words[i] in _NUMBER_WORDS:
                n = _NUMBER_WORDS[words[i]]
                if n == 100:
                    current_group = max(current_group, 1) * 100
                elif n == 1000:
                    val += max(current_group, 1) * 1000
                    current_group = 0
                elif n < 100:
                    # Handle "six fifty" (6 50) -> 650
                    if current_group > 0 and n >= 10:
                        val += current_group * 100 + n
                        current_group = 0
                    elif current_group > 0 and n < 10 and current_group < 10:
                        # "six five zero" -> 650
                        current_group = current_group * 10 + n
                    else:
                        current_group += n
                i += 1
            val += current_group
            out.append(str(val))
        else:
            out.append(w)
            i += 1
    return " ".join(out)

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return words_to_numbers(text)


# ── Globals (loaded at startup) ────────────────────────────────────────────────
_index: Optional[faiss.Index]        = None
_meta:  Optional[list]               = None
_brand_names: list[str]              = []
_tree                                = None   # BKTree instance
_model: Optional[SentenceTransformer] = None


app = FastAPI(title="Medicine Search API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def load_artifacts() -> None:
    global _index, _meta, _tree, _model, _brand_names

    print(f"[Startup] Loading FAISS index from {INDEX_PATH} …")
    if not INDEX_PATH.exists():
        print("  WARNING: FAISS index not found. Run steps 1-3 first.")
    else:
        _index = faiss.read_index(str(INDEX_PATH))
        # Increase efSearch for better recall
        if hasattr(_index, "hnsw"):
            _index.hnsw.efSearch = 64
        print(f"  FAISS loaded. Vectors: {_index.ntotal:,}")

    print(f"[Startup] Loading medicine metadata from {META_PATH} …")
    if not META_PATH.exists():
        print("  WARNING: medicine_meta.json not found.")
    else:
        with open(META_PATH, encoding="utf-8") as f:
            _meta = json.load(f)
        _brand_names = [m["brand_name"].lower() for m in _meta]
        print(f"  Meta loaded. Entries: {len(_meta):,}")

    print(f"[Startup] Loading BK-Tree from {TREE_PATH} …")
    if not TREE_PATH.exists():
        print("  WARNING: bktree.pkl not found.")
    else:
        with open(TREE_PATH, "rb") as f:
            _tree = pickle.load(f)
        print("  BK-Tree loaded.")

    print(f"[Startup] Loading sentence-transformer model: {MODEL_NAME} …")
    try:
        _model = SentenceTransformer(MODEL_NAME)
        print("  Model loaded. Server ready.")
    except Exception as exc:
        # Keep the service usable offline. Fuzzy and BK-tree matching still work
        # even when the embedding model cannot be fetched from Hugging Face.
        _model = None
        print(f"  WARNING: model load failed ({exc.__class__.__name__}: {exc}).")
        print("  Continuing with fuzzy + BK-tree matching only.")


def levenshtein(s1: str, s2: str) -> int:
    if s1 == s2: return 0
    if not s1: return len(s2)
    if not s2: return len(s1)
    m, n = len(s1), len(s2)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]

class BKNode:
    __slots__ = ("word", "children")
    def __init__(self, word: str):
        self.word = word
        self.children: dict[int, "BKNode"] = {}
    def insert(self, word: str) -> None:
        d = levenshtein(self.word, word)
        if d == 0: return
        if d in self.children: self.children[d].insert(word)
        else: self.children[d] = BKNode(word)
    def search(self, query: str, max_dist: int) -> list[tuple[int, str]]:
        d = levenshtein(self.word, query)
        results = []
        if d <= max_dist: results.append((d, self.word))
        for k, child in self.children.items():
            if abs(k - d) <= max_dist:
                results.extend(child.search(query, max_dist))
        return results

class BKTree:
    def __init__(self) -> None:
        self._root: Optional[BKNode] = None
    def insert(self, word: str) -> None:
        if self._root is None: self._root = BKNode(word)
        else: self._root.insert(word)
    def search(self, query: str, max_dist: int = 2) -> list[tuple[int, str]]:
        if self._root is None: return []
        return sorted(self._root.search(query, max_dist))

class SearchRequest(BaseModel):
    query: str
    top_k: int = FAISS_TOP_K


class MedicineMeta(BaseModel):
    id: int
    brand_name: str
    generic_name: str
    strength: str
    form: str


class SearchResponse(BaseModel):
    query: str
    normalized_query: str
    match: Optional[str]
    score: float
    confidence: str        # "high" | "low" | "bktree" | "none"
    medicine: Optional[MedicineMeta]
    elapsed_ms: float


# ── Phonetic / STT alias table ────────────────────────────────────────────────
# Maps normalized mis-heard phrases → correct brand name (not in DB).
_ALIASES: dict[str, str] = {
    "cal coral dee": "Kalcoral D",
    "calculate d":   "Kalcoral D",
    "kal coral d":   "Kalcoral D",
    "karakural d":   "Kalcoral D",
    "karakural":     "Kalcoral",
    "ravi prasoon":  "Rabeprazole",
    "ravi prasual":  "Rabeprazole",
    "ravi prasal":   "Rabeprazole",
    "ravi prasul":   "Rabeprazole",
    "ravi prazool":  "Rabeprazole",
    "happy prasul":  "Rabeprazole",
    "happy prasoon": "Rabeprazole",
    "happy prasal":  "Rabeprazole",
    "happi prasul":  "Rabeprazole",
    "happi prasoon": "Rabeprazole",
    "brazil brazil": "Rabeprazole",
    "baby brazil":   "Rabeprazole",
    "baby prasoon":  "Rabeprazole",
    "baby prasul":   "Rabeprazole",
    "baby prasal":   "Rabeprazole",
    "enterprises":   "Pantoprazole",
}

# ── Candidate dataclass ───────────────────────────────────────────────────────
from dataclasses import dataclass

@dataclass
class _Candidate:
    score: float
    meta: Optional[dict]
    match_name: str
    source: str  # "fuzzy" | "faiss" | "bktree" | "alias"


# ── Three engines, always all run ─────────────────────────────────────────────

def _run_fuzzy(query_norm: str) -> Optional[_Candidate]:
    if not _meta:
        return None
    hit = process.extractOne(query_norm, _brand_names, scorer=fuzz.token_set_ratio)
    if hit:
        name, score, idx = hit
        return _Candidate(score=score, meta=_meta[idx], match_name=_meta[idx]["brand_name"], source="fuzzy")
    return None


def _run_faiss(query_norm: str, top_k: int) -> Optional[_Candidate]:
    if _model is None or _index is None or _meta is None:
        return None
    q_vec = _model.encode([query_norm], normalize_embeddings=True).astype("float32")
    distances, indices = _index.search(q_vec, top_k)
    best_score, best_meta = 0.0, None
    for idx in indices[0]:
        if idx < 0 or idx >= len(_meta):
            continue
        m = _meta[idx]
        score = fuzz.token_set_ratio(query_norm, normalize(m["brand_name"]))
        if score > best_score:
            best_score, best_meta = score, m
    if best_meta:
        return _Candidate(score=best_score, meta=best_meta, match_name=best_meta["brand_name"], source="faiss")
    return None


def _run_bktree(query_norm: str) -> Optional[_Candidate]:
    """
    Searches the BK-tree against every word and every 1-2 word chunk of the
    query, so multi-word brand names like 'Kalcoral D' are reachable.
    """
    if _tree is None or _meta is None:
        return None
    words = query_norm.split()
    # Build chunks: single words + bigrams
    chunks = words + [" ".join(words[i:i+2]) for i in range(len(words)-1)]
    best: Optional[_Candidate] = None
    for chunk in chunks:
        results = _tree.search(chunk, max_dist=BKTREE_MAX_DIST)
        for dist, bk_name in results:
            bk_score = max(0.0, 100.0 - dist * 15)
            if bk_score < 55.0:
                continue
            m = next((x for x in _meta if x["brand_name"].lower() == bk_name), None)
            if best is None or bk_score > best.score:
                best = _Candidate(score=bk_score, meta=m,
                                  match_name=m["brand_name"] if m else bk_name.title(),
                                  source="bktree")
    return best


# ── Endpoint ──────────────────────────────────────────────────────────────────
@app.post("/search", response_model=SearchResponse)
def search_medicine(req: SearchRequest) -> SearchResponse:
    t0 = time.time()
    query_norm = normalize(req.query)

    if not query_norm:
        raise HTTPException(status_code=400, detail="Empty query")

    # 0. Alias table — hardcoded for severe phonetic hallucinations
    if query_norm in _ALIASES:
        alias_name = _ALIASES[query_norm]
        alias_meta = next((m for m in (_meta or []) if m["brand_name"].lower() == alias_name.lower()), None)
        return SearchResponse(
            query=req.query, normalized_query=query_norm,
            match=alias_meta["brand_name"] if alias_meta else alias_name,
            score=100.0, confidence="high",
            medicine=MedicineMeta(**alias_meta) if alias_meta else None,
            elapsed_ms=round((time.time() - t0) * 1000, 1),
        )

    # 1. Run ALL three engines unconditionally
    fuzzy_c  = _run_fuzzy(query_norm)
    faiss_c  = _run_faiss(query_norm, req.top_k)
    bktree_c = _run_bktree(query_norm)

    # 2. Pick the highest-scoring candidate across all three
    candidates = [c for c in (fuzzy_c, faiss_c, bktree_c) if c is not None]
    if not candidates:
        return SearchResponse(query=req.query, normalized_query=query_norm,
                              match=None, score=0.0, confidence="none",
                              medicine=None, elapsed_ms=round((time.time() - t0) * 1000, 1))

    best = max(candidates, key=lambda c: c.score)

    confidence = "high" if best.score >= CONFIDENCE_THRESHOLD else \
                 "bktree" if best.source == "bktree" else "low"

    return SearchResponse(
        query=req.query,
        normalized_query=query_norm,
        match=best.match_name,
        score=best.score,
        confidence=confidence,
        medicine=MedicineMeta(**best.meta) if best.meta else None,
        elapsed_ms=round((time.time() - t0) * 1000, 1),
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "index_loaded": _index is not None,
        "meta_loaded": _meta is not None,
        "tree_loaded": _tree is not None,
        "model_loaded": _model is not None,
        "total_medicines": _index.ntotal if _index else 0,
    }


if __name__ == "__main__":
    import uvicorn
    # When run directly, __main__ is this module, so BKTree will unpickle correctly.
    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
