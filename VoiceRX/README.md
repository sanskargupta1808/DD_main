# VoiceRX

Transcribe doctor–patient conversations and automatically extract the clinically
important details into a structured summary:

- **Symptoms** mentioned by the patient
- **Diagnoses / diseases**
- **Prescriptions** — medication, dosage (amount), frequency/timing, duration, instructions
- **Follow-up** — next visit date and instructions
- Free-text **notes**

## How it works

```
 ┌─────────────┐   mic (MediaRecorder) ──▶ audio recording (playback/download)
 │   Browser   │
 │  (client)   │ ─── Web Speech API ────▶ live transcript (real time)
 │             │ ── transcript text ────▶ POST /api/extract
 └─────────────┘                                  │
                                         ┌─────────▼────────────┐
                                         │  server (Express/TS)  │
                                         │  extraction service   │
                                         │  • heuristic (default)│
                                         │  • OpenAI / Bedrock   │
                                         └─────────┬────────────┘
                                                   │ structured JSON
                                         ┌─────────▼────────────┐
                                         │  Extraction view UI   │
                                         └──────────────────────┘
```

- **Recording + accurate transcription** — pressing **Record** captures the mic
  audio (MediaRecorder). On **Stop**, the recording is sent to `POST /api/transcribe`
  and transcribed by **Groq Whisper (`whisper-large-v3`)** with a medical-vocabulary
  prompt — far more accurate on drug names and Indian accents than the browser. The
  browser Web Speech API is used only as an optional live preview while recording.
- **Extraction** posts the transcript to `POST /api/extract`. By default it uses a
  built-in rule-based extractor so the app works with zero configuration. Set
  `EXTRACTION_PROVIDER=groq` (uses `GROQ_API_KEY`), `openai`, or `bedrock` for
  higher-quality LLM extraction.
- **STT correction (optional)** — before extraction, `/api/extract` can clean the
  transcript: fix misheard Indian medicine names, normalise vitals/dosage phrasing.
  Ported from the Flutter Voice Rx stack (see "Transcript correction" below).

## Project layout

```
VoiceRX/
├── server/             Express + TypeScript API (correction + extraction)
├── client/             React + Vite + TypeScript UI (recorder, transcript, results)
├── medicine_pipeline/  Python FastAPI FAISS medicine-search service (268K Indian drugs)
└── shared types are duplicated in server/src/types.ts and client/src/types.ts
   (kept identical — update both if you change the schema)
```

## Getting started

```bash
# from the VoiceRX root
cp .env.example .env        # optional: customise providers/keys
npm install                 # installs both workspaces

npm run dev                 # starts server (:4000) and client (:5173)
```

Open http://localhost:5173, click **Record**, and speak — the conversation is
recorded and transcribed live (or paste/type a transcript). Click **Extract** to
generate the structured summary, and play back or download the recording.

## Language & transcription provider

Pick the spoken language from the dropdown in the UI (Auto-detect, English (India),
Hindi, Gujarati, Marathi, Bengali, Tamil, Telugu, Kannada, Malayalam, Odia, Punjabi).
The chosen locale is sent with the audio and:
- sets the STT `language`, and
- sets the Indian-OPD correction prompt's language.

Transcription is pluggable via `TRANSCRIPTION_PROVIDER`:
- `groq` (default) — Groq Whisper `whisper-large-v3`. Good for English; weaker on
  many Indic languages.
- `bhashini` — **free** Govt-of-India Bhashini/ULCA ASR, which runs **AI4Bharat
  IndicConformer** (strong on the 22 Indian languages). Setup:
  1. Register at <https://bhashini.gov.in> (Bhashini Udyat) and copy your
     **userID** and **ulcaApiKey** from *My Profile*.
  2. In `.env`: `TRANSCRIPTION_PROVIDER=bhashini`, `BHASHINI_USER_ID=…`,
     `BHASHINI_ULCA_API_KEY=…` (optionally `BHASHINI_PIPELINE_ID`).
  3. Requires **ffmpeg** installed on the server (audio is transcoded to 16 kHz
     mono WAV before upload). Then restart and pick the language in the UI.
  Notes: Bhashini ASR needs a language (Auto-detect falls back to Hindi), and it
  returns text in the native script — the Groq correction/extraction handle that.
- `custom` — any OpenAI-compatible `/audio/transcriptions` server, e.g. a
  **self-hosted / fine-tuned IndicWhisper** behind `faster-whisper-server` or
  `speaches`. Set `TRANSCRIPTION_BASE_URL`, `TRANSCRIPTION_MODEL`, and
  (optionally) `TRANSCRIPTION_API_KEY`.

In all cases the rest of the pipeline (correction + extraction) is unchanged.

### Dual-ASR (regional conversation + English medicine names)

Set `DUAL_ASR=true` (with a non-`groq` regional provider + a `GROQ_API_KEY`) to run
**two engines in parallel on the same recording** and merge them:
- the **regional** provider (IndicConformer/Bhashini/custom) transcribes the
  conversation accurately in the spoken language, and
- **Groq Whisper (→English)** produces an English transcript, from which the
  medicine names are extracted and resolved against the **FAISS drug DB** (the DB
  is Latin-script, so the English transcript is what makes drug-name matching work).

Groq then merges them into one transcript: the conversation stays in the regional
language/script, but **every medicine name is rendered in English**. This fixes the
common case where the regional STT mangles a drug name that Whisper catches in
English. Trade-off: two STT calls per recording (more latency/cost). The merged
transcript is already corrected, so extraction skips its correction pass.

### Bundled local IndicConformer (free, offline, no API key)

`indic_stt/` is a small FastAPI service that wraps **AI4Bharat IndicConformer-600M
`indic_stt/` is a small FastAPI service that wraps **AI4Bharat IndicConformer-600M**
and exposes the OpenAI-compatible `/audio/transcriptions` endpoint — so it plugs
into the `custom` provider above. It runs **CPU-only, needs no GPU, no HF token,
and no per-use cost**. Requires **ffmpeg** on the server.

**Precision (`INDIC_PRECISION`, default `fp32`):**
- `fp32` — full-precision model from an ungated mirror
  (`sunilmahendrakar/indic-conformer-600m-multilingual`). Much better accuracy,
  especially for non-Hindi languages. ~2.5 GB download, CTC decoding, slower on CPU.
- `int8` — quantized model (`indic-asr-onnx`, ~730 MB). Smaller/faster but ~2× WER;
  really only good for Hindi.

```bash
cd indic_stt
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
INDIC_PRECISION=fp32 .venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8001
```

Then point VoiceRX at it (already set in `.env` for this project):

```
TRANSCRIPTION_PROVIDER=custom
TRANSCRIPTION_BASE_URL=http://localhost:8001
TRANSCRIPTION_MODEL=indic-conformer-600m
```

Covers all 22 Indian languages (Hindi, Gujarati, Marathi, Bengali, Tamil, Telugu,
Kannada, Malayalam, Odia, Punjabi, …). Notes: it's **Indic-only** — English falls
back to Hindi, so use the `groq` provider (or dual mode) for English; the first
request loads the model (fp32 takes ~20 s on CPU, then it's cached).

## Speaker diarization ("Label Doctor/Patient")

`POST /api/diarize` labels who said what, via three interchangeable modes
(`mode` form field):

- **`ai`** — no audio, no acoustic detection at all. Sends the plain
  transcript text to Groq and asks it to *guess* speaker turns from
  conversational context (doctor asks, patient answers). Fast and free of
  audio upload, but it's a pure language-model guess — it can mislabel a
  turn whose content doesn't clearly read as one role or the other.
- **`hybrid`** — real acoustic detection (`indic_stt`'s resemblyzer + KMeans
  voice-embedding clustering) finds speaker-change timestamps, but the
  transcript text is then distributed across those timestamps *proportionally
  by character position* (`frequencyDiarize.ts`'s `applySegments`) — a
  best-effort guess, not a real alignment, since the STT step produces no
  word-level timestamps. This proportional guess is what actually becomes the
  `diarized` output; Groq is then asked to relabel the resulting Speaker
  A/B → Doctor/Patient. Independently-transcribed per-speaker containers are
  also returned, but they aren't chronologically interleaved and don't feed
  the main `diarized` string. In practice the proportional split can misattribute
  whole sentences, especially near the start of a turn.
- **`acoustic`** — real acoustic detection via `indic_stt`'s **pyannote**
  neural pipeline (`pyannote/speaker-diarization-3.1`; requires `HF_TOKEN`,
  see below), aligned against **real word-level timestamps** from a single
  whole-recording Groq Whisper transcription — **no LLM anywhere in the
  path, and no guessing of any kind**: every word is attributed to whichever
  acoustic speaker segment is active at that word's actual timestamp.
  Output is chronologically interleaved and labeled generically
  `User 1:`/`User 2:` (first-appearance order), not a Doctor/Patient role
  guess. Uses `/audio/transcriptions` (not `/translations`), so multilingual/
  code-switched speech comes back in its original language/script — verified
  correctly interleaving Hindi and English within one conversation.
  (An earlier version of this mode transcribed each detected turn in
  isolation instead. Don't do that: handed a bare 1-2s clip of a quick
  back-and-forth exchange with no surrounding context, both Whisper and
  IndicConformer reliably hallucinate fluent nonsense, e.g. "subscribe to
  our channel." A single whole-recording pass has the context to be
  reliable — proved by comparing against the exact same audio's normal
  `/api/transcribe` output.) Trade-off: doesn't attempt to say *which*
  speaker is the doctor, and needs `GROQ_API_KEY` (word timestamps aren't
  available from the `custom`/IndicConformer/Bhashini providers).

Setup for `acoustic` mode — pyannote's models are free but gated on Hugging Face:
1. Create a token at <https://huggingface.co/settings/tokens> (read access is enough).
2. Accept the license (one click each) on:
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/segmentation-3.0>
   - <https://huggingface.co/pyannote/speaker-diarization-community-1>
     (a scoring dependency pulled in by pyannote.audio 4.x — easy to miss)
3. In `.env`: `HF_TOKEN=hf_...`.
4. `pip install -r indic_stt/requirements.txt` again if `indic_stt/.venv` predates
   this — it now also installs `torch`, `torchaudio`, and `pyannote.audio`.

## Transcript correction (medicine names, vitals, dosage)

Browser STT often mishears Indian brand drugs ("cal coral dee" instead of
"Kalcoral D"). VoiceRX ports the correction stack from the Flutter Voice Rx app.
When `/api/extract` runs with `correct: true` (default, `CORRECTION_ENABLED=true`),
the transcript goes through this pipeline before extraction:

1. **Alias pre-scan** — hardcoded phonetic hallucinations (works with no API key).
2. **Groq medicine NER** — `llama-3.1-8b-instant` extracts medicine tokens.
3. **FAISS normalization** — each token is resolved against the 268K-drug
   medicine-search service (`medicine_pipeline/`).
4. **Groq cleanup** — Indian-OPD prompt normalises vitals ("one oh one" → "101"),
   dosage ("six fifty mg" → "650 mg"), and applies the medicine corrections.

Every stage degrades gracefully: with no `GROQ_API_KEY` only the alias table runs;
if the medicine service is down, that step is skipped. The Groq key lives in
server env only — the browser never sees it.

### 1. Configure the Groq key

```bash
# in .env
CORRECTION_ENABLED=true
GROQ_API_KEY=gsk_...          # from https://console.groq.com
MEDICINE_SEARCH_URL=http://localhost:8000
```

### 2. Start the medicine-search service (Python)

The 268K-medicine FAISS index/meta/BK-tree artifacts live in
`medicine_pipeline/output/`. Use Python 3.10–3.12 (heavy ML wheels):

```bash
python3.12 -m venv medicine_pipeline/.venv
medicine_pipeline/.venv/bin/pip install -r medicine_pipeline/requirements.txt
# run as a script (so the BK-tree pickle resolves correctly):
cd medicine_pipeline && .venv/bin/python server.py   # serves :8000
```

Check it: `curl -X POST localhost:8000/search -d '{"query":"cal coral dee"}' -H 'Content-Type: application/json'`
→ `{"match":"Kalcoral D","score":100,"confidence":"high"}`.

The endpoints `POST /api/correct` (correction only) and `GET /api/health`
(reports correction status) are also available.

Other scripts:

```bash
npm run build       # type-check + build both packages
npm run typecheck   # type-check only
```

## ⚠️ Security, privacy & compliance

This app handles Protected Health Information (PHI). Before any real-world use:

- **The API is currently unauthenticated.** Add authentication/authorization and
  per-user access controls before deploying anywhere reachable.
- Serve over **HTTPS/TLS** and encrypt any stored audio/transcripts at rest.
- Audio, transcripts, and extracted data are **never written to the repo**
  (see `.gitignore`), and nothing is persisted by default.
- If you enable a cloud provider (OpenAI/Bedrock), patient data is sent to that
  third party — ensure you have a **BAA / appropriate data-processing agreement**
  and comply with HIPAA/GDPR/local regulations.
- This tool is a documentation aid. It does **not** provide medical advice and its
  output must be reviewed by a qualified clinician.
