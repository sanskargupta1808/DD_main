# Medicine FAISS Pipeline

End-to-end pipeline that converts 268K Indian medicines from Excel into a searchable FAISS vector index used for real-time medicine name correction in the Voice Rx feature.

## Architecture

```
Flutter STT Transcript
        ↓
Medical Entity Detection
        ↓
Candidate Medicine Token
        ↓
Normalization
        ↓
FAISS Vector Search  ──→  Top 20 Candidates
        ↓
RapidFuzz Re-ranking
        ↓
Confidence Check
  ├── score ≥ 85 → Accept
  └── score < 85 → BK-Tree Validation
                        ↓
                  Final Match
```

## Files

| File | Purpose |
|---|---|
| `1_excel_to_json.py` | Excel → `medicines.json` + `medicine_meta.json` |
| `2_generate_embeddings.py` | JSON → `medicine_embeddings.npy` |
| `3_build_faiss_index.py` | Embeddings → `medicine.index` |
| `4_build_bktree.py` | Brand names → `bktree.pkl` |
| `server.py` | FastAPI search server |
| `requirements.txt` | Python dependencies |
| `output/` | Generated artefacts (gitignored) |

## Setup (one-time)

```bash
# 1. Install dependencies
pip3 install -r tools/medicine_pipeline/requirements.txt

# 2. Convert Excel → JSON (~1 min)
python3 tools/medicine_pipeline/1_excel_to_json.py

# 3. Generate embeddings (~15-25 min on CPU, ~2-4 min on GPU)
python3 tools/medicine_pipeline/2_generate_embeddings.py

# 4. Build FAISS index (~2 min)
python3 tools/medicine_pipeline/3_build_faiss_index.py

# 5. Build BK-Tree fallback (~3 min)
python3 tools/medicine_pipeline/4_build_bktree.py
```

## Running the Server

```bash
# From repo root:
python3 -m uvicorn tools.medicine_pipeline.server:app --host 0.0.0.0 --port 8000

# Or with auto-reload during development:
python3 -m uvicorn tools.medicine_pipeline.server:app --host 0.0.0.0 --port 8000 --reload
```

## Testing

```bash
# Health check
curl http://localhost:8000/health

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "cal coral dee"}'

# Expected: {"match": "Kalcoral D", "score": ~92, "confidence": "high"}
```

## Flutter Configuration

Set the server URL once in your app (defaults to `http://localhost:8000`):

```dart
await MedicineSearchService.setServerUrl('http://192.168.1.10:8000');
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PIPELINE_DIR` | `tools/medicine_pipeline/output` | Path to artefacts |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CONFIDENCE_THRESHOLD` | `85.0` | Min RapidFuzz score to accept |
| `FAISS_TOP_K` | `20` | Candidates from FAISS |
| `BKTREE_MAX_DIST` | `3` | Max Levenshtein distance |
