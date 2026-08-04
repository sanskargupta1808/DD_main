# VoiceRX Flask API

A Python/Flask API that exposes the full VoiceRX pipeline. It orchestrates the
existing services — **IndicConformer STT** (`indic_stt`, :8001), **medicine FAISS
search** (`medicine_pipeline`, :8000) — plus **Groq** (Whisper + LLMs).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/api/health` | Service status + configured models |
| POST | `/api/record` | Upload + store an audio recording → `{audio_id, download_url}` |
| GET  | `/api/audio/<audio_id>` | Download the stored audio |
| POST | `/api/transcribe` | All-stage transcription → `{regional, english, final, corrections, language, dual}` |
| POST | `/api/extract` | Extract structured data from a transcript → `{result_id, extraction, download_url}` |
| GET  | `/api/result/<result_id>` | Fetch a stored extraction (JSON) |
| GET  | `/api/result/<result_id>/download` | Download the extraction as a `.json` file |

### Behaviour
- **Transcription stages** (`/api/transcribe`): for a non-English locale it runs
  IndicConformer (regional) **and** Groq Whisper (English) and merges them with Groq
  acting as a medical transcriptionist — conversation stays in the selected language,
  medicine names in English. For `locale=en_IN` it returns a single English transcript.
- **Extraction** (`/api/extract`): structured JSON with all fields in English; the exact
  drug spoken is kept (no brand↔generic substitution).

`/api/transcribe` accepts either a direct `audio` file upload (multipart) or a stored
`audio_id` (from `/api/record`), plus a `locale` (e.g. `en_IN`, `hi_IN`, `gu_IN`).

## Setup & run

```bash
cd flask_api
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py        # serves on :5005 (override with FLASK_PORT)
```

Prerequisites (started separately):
- IndicConformer STT: `cd ../indic_stt && .venv/bin/python -m uvicorn server:app --port 8001`
- Medicine FAISS: `cd ../medicine_pipeline && .venv/bin/python server.py`  (:8000)
- `GROQ_API_KEY` in the repo-root `.env`

Config (read from the repo-root `.env`): `GROQ_API_KEY`, `GROQ_MODEL`,
`GROQ_EXTRACTION_MODEL`, `GROQ_TRANSCRIPTION_MODEL`, `INDIC_STT_URL`,
`MEDICINE_SEARCH_URL`, `CORRECTION_LOCALE`, `FLASK_PORT`.

## Examples

```bash
# record → audio_id
curl -X POST localhost:5005/api/record -F "audio=@consult.m4a"

# all-stage transcription
curl -X POST localhost:5005/api/transcribe -F "audio=@consult.m4a" -F "locale=hi_IN"

# extract from the final transcript
curl -X POST localhost:5005/api/extract -H "Content-Type: application/json" \
  -d '{"transcript":"…","locale":"hi_IN"}'

# download the extraction
curl -OJ localhost:5005/api/result/<result_id>/download
```

## ⚠️ Notes
- Stored audio + results live in `flask_api/storage/` (gitignored — they are PHI).
- This dev server is **unauthenticated** and uses Flask's dev WSGI server. Add
  authentication + a production WSGI server (gunicorn/uvicorn) before any real use.
