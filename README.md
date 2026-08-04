# VoiceRX + DoctorDiary

A combined repo for two clinical-documentation projects that run together:

- **`VoiceRX/`** — the speech-to-text, dual-ASR (regional + English), medical
  extraction, and speaker-diarization backend/frontend.
- **`DoctorDiary_main/`** — the patient-record/consultation web app, which
  hands off audio recordings to VoiceRX for transcription and diarization.

The DoctorDiary Flutter apps are **not** included in this repo.

## Prerequisites

- Python 3.12+ (each Python subproject gets its own virtualenv — see below)
- Node.js 18+ and npm (for `VoiceRX/server` and `VoiceRX/client`)
- `ffmpeg` on your `PATH`
- A [Groq](https://console.groq.com) API key
- A [Hugging Face](https://huggingface.co/settings/tokens) token, with the
  license accepted for `pyannote/speaker-diarization-3.1`,
  `pyannote/segmentation-3.0`, and `pyannote/speaker-diarization-community-1`
  (only needed for pyannote-based diarization in `indic_stt`)

Works cross-platform (macOS, Linux, Windows) — the start scripts are plain
Python, no bash/WSL required.

## First-time setup

```bash
# VoiceRX
cd VoiceRX
cp .env.example .env        # then fill in GROQ_API_KEY, HF_TOKEN, etc.
python3 -m venv indic_stt/.venv && indic_stt/.venv/bin/pip install -r indic_stt/requirements.txt
python3 -m venv medicine_pipeline/.venv && medicine_pipeline/.venv/bin/pip install -r medicine_pipeline/requirements.txt
python3 -m venv flask_api/.venv && flask_api/.venv/bin/pip install -r flask_api/requirements.txt
npm install

# DoctorDiary
cd ../DoctorDiary_main
cp .env.example .env        # then fill in GROQ_API_KEY
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

(On Windows, venv binaries live under `.venv\Scripts\` instead of
`.venv/bin/` — adjust the paths above accordingly.)

### Medicine search data (not included)

`VoiceRX/medicine_pipeline/output/` (the FAISS index + metadata used for
medicine-name lookup) is excluded from this repo — the index file alone is
464 MB, well over GitHub's 100 MB file limit, and its source spreadsheet
wasn't available when this repo was assembled. To regenerate it, you need a
medicine database spreadsheet and to run, in order:

```bash
cd VoiceRX/medicine_pipeline
.venv/bin/python 1_excel_to_json.py
.venv/bin/python 2_generate_embeddings.py
.venv/bin/python 3_build_faiss_index.py
.venv/bin/python 4_build_bktree.py
```

Without this, everything else works — medicine-name auto-correction during
transcription just won't have a lookup index to draw on.

## Running it

```bash
# Everything (VoiceRX's own services + Node/React):
cd VoiceRX && python3 start.py

# Everything, including DoctorDiary:
cd DoctorDiary_main && python3 start_project.py
```

`start_project.py` looks for VoiceRX as a sibling directory (`../VoiceRX`
relative to `DoctorDiary_main/`, i.e. this repo's layout works out of the
box) — set `VOICERX_DIR` if you've moved it elsewhere.

Default ports: DoctorDiary `8080`, VoiceRX Flask API `5005`, VoiceRX Node API
`4000`, VoiceRX React UI `5173`, IndicConformer STT `8001`, medicine search
`8000`.

## What's deliberately not in this repo

- `.env` files (secrets) — use the `.env.example` templates
- Any Google service-account credential JSON
- SQLite databases and uploaded audio (real patient/consultation data)
- `node_modules/`, `.venv/`, build output — reinstall/rebuild locally
- `medicine_pipeline/output/` — see above
- The DoctorDiary Flutter apps — out of scope for this repo
