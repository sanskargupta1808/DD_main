#!/usr/bin/env python3
"""
Step 4: Build a BK-Tree from all medicine brand names (Levenshtein fallback).
Run after 1_excel_to_json.py:
    python3 tools/medicine_pipeline/4_build_bktree.py

Output:
    tools/medicine_pipeline/output/bktree.pkl
"""
import json
import pickle
import sys
import time
from pathlib import Path

IN_META  = Path("tools/medicine_pipeline/output/medicine_meta.json")
OUT_TREE = Path("tools/medicine_pipeline/output/bktree.pkl")


# ── Pure-Python BK-Tree (no external dependency) ─────────────────────────────

def levenshtein(s1: str, s2: str) -> int:
    """Levenshtein edit distance."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    m, n = len(s1), len(s2)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,       # insertion
                prev[j] + 1,           # deletion
                prev[j - 1] + cost,    # substitution
            )
        prev = curr
    return prev[n]


class BKNode:
    __slots__ = ("word", "children")

    def __init__(self, word: str):
        self.word = word
        self.children: dict[int, "BKNode"] = {}

    def insert(self, word: str) -> None:
        d = levenshtein(self.word, word)
        if d == 0:
            return
        if d in self.children:
            self.children[d].insert(word)
        else:
            self.children[d] = BKNode(word)

    def search(self, query: str, max_dist: int) -> list[tuple[int, str]]:
        d = levenshtein(self.word, query)
        results: list[tuple[int, str]] = []
        if d <= max_dist:
            results.append((d, self.word))
        for k, child in self.children.items():
            if abs(k - d) <= max_dist:
                results.extend(child.search(query, max_dist))
        return results


class BKTree:
    def __init__(self) -> None:
        self._root: BKNode | None = None

    def insert(self, word: str) -> None:
        if self._root is None:
            self._root = BKNode(word)
        else:
            self._root.insert(word)

    def search(self, query: str, max_dist: int = 2) -> list[tuple[int, str]]:
        if self._root is None:
            return []
        return sorted(self._root.search(query, max_dist))


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"[1/4] Loading medicine metadata from {IN_META} …")
    with open(IN_META, encoding="utf-8") as f:
        meta = json.load(f)

    # Work on lowercase brand names for case-insensitive matching
    brand_names = [m["brand_name"].lower() for m in meta]
    print(f"      {len(brand_names):,} brand names loaded.")

    print(f"[2/4] Building BK-Tree (this may take a few minutes) …")
    tree = BKTree()
    t0 = time.time()
    for i, name in enumerate(brand_names, 1):
        tree.insert(name)
        if i % 50_000 == 0:
            print(f"      … inserted {i:,} / {len(brand_names):,}")
    elapsed = time.time() - t0
    print(f"      Built in {elapsed:.1f}s.")

    print(f"[3/4] Saving BK-Tree to {OUT_TREE} …")
    OUT_TREE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TREE, "wb") as f:
        pickle.dump(tree, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = OUT_TREE.stat().st_size / 1e6
    print(f"      Saved. File size: {size_mb:.1f} MB")

    # Quick sanity check
    print(f"[4/4] Sanity check — searching 'dolo' with max_dist=2 …")
    results = tree.search("dolo 650 tablet", max_dist=2)
    print(f"      Matches: {results[:5]}")
    print("      Done!")


if __name__ == "__main__":
    main()
