#!/usr/bin/env python3
"""
VoiceRX Flask API.

Endpoints
  GET  /api/health                      service status
  POST /api/record                      upload + store an audio recording  -> {audio_id}
  GET  /api/audio/<audio_id>            download the stored audio
  POST /api/transcribe                  audio (file or audio_id) + locale   -> all stages
                                        {regional, english, final, corrections, language}
  POST /api/extract                     {transcript, locale}                -> {result_id, extraction}
  POST /api/session                     save audio + all existing stage results -> {session_id}
  GET  /api/session/<session_id>        fetch a saved DoctorDiary handoff
  POST /api/diarize/retranscribe        retry one speaker's transcript after a hybrid diarize
  GET  /api/result/<result_id>          fetch a stored extraction (JSON)
  GET  /api/result/<result_id>/download download the extraction as a .json file

Run:
    .venv/bin/python -m flask --app app run --port 5005
  or
    .venv/bin/python app.py
"""
import json
import time
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file
from flask_cors import CORS

import pipeline

STORAGE = Path(__file__).resolve().parent / "storage"
AUDIO_DIR = STORAGE / "audio"
RESULT_DIR = STORAGE / "results"
SESSION_DIR = STORAGE / "sessions"
for d in (AUDIO_DIR, RESULT_DIR, SESSION_DIR):
    d.mkdir(parents=True, exist_ok=True)

MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25 MB
ALLOWED_EXT = {".webm", ".m4a", ".mp4", ".ogg", ".wav", ".mp3", ".flac"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)


def _safe_id(raw: str) -> str:
    """Reject anything that isn't a plain uuid-hex token (path-traversal guard)."""
    if not raw or not all(c.isalnum() or c in "-" for c in raw):
        abort(400, "invalid id")
    return raw


def _ext_for(filename: str, mimetype: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in ALLOWED_EXT:
        return suffix
    if "mp4" in mimetype or "m4a" in mimetype:
        return ".m4a"
    if "ogg" in mimetype:
        return ".ogg"
    if "wav" in mimetype:
        return ".wav"
    if "mpeg" in mimetype or "mp3" in mimetype:
        return ".mp3"
    return ".webm"


def _find_audio(audio_id: str) -> Path:
    for p in AUDIO_DIR.glob(f"{audio_id}.*"):
        return p
    abort(404, "audio not found")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "groqConfigured": bool(pipeline.GROQ_API_KEY),
        "indicSttUrl": pipeline.INDIC_STT_URL,
        "medicineSearchUrl": pipeline.MEDICINE_SEARCH_URL,
        "models": {
            "transcription": pipeline.GROQ_TRANSCRIPTION_MODEL,
            "ner": pipeline.GROQ_NER_MODEL,
            "extraction": pipeline.GROQ_EXTRACTION_MODEL,
        },
        "defaultLocale": pipeline.DEFAULT_LOCALE,
    })


@app.post("/api/record")
def record():
    """Store an uploaded audio recording. Field name: 'audio'."""
    f = request.files.get("audio")
    if not f:
        abort(400, "audio file (field 'audio') is required")
    audio_id = uuid.uuid4().hex
    ext = _ext_for(f.filename, f.mimetype or "")
    path = AUDIO_DIR / f"{audio_id}{ext}"
    f.save(path)
    return jsonify({
        "audio_id": audio_id,
        "filename": f"{audio_id}{ext}",
        "mimetype": f.mimetype,
        "size_bytes": path.stat().st_size,
        "download_url": f"/api/audio/{audio_id}",
    })


@app.get("/api/audio/<audio_id>")
def download_audio(audio_id):
    path = _find_audio(_safe_id(audio_id))
    return send_file(path, as_attachment=True, download_name=path.name)


@app.post("/api/session")
def create_session():
    """Persist an already-processed consultation for the React VoiceRX UI.

    DoctorDiary uploads the audio through /api/record first, then sends the
    regional/Whisper/final stages and extraction here. This keeps the handoff
    small and lets VoiceRX load the original recording from its own storage.
    """
    body = request.get_json(silent=True) or {}
    audio_id = body.get("audio_id")
    if not audio_id:
        abort(400, "field 'audio_id' is required")
    audio_path = _find_audio(_safe_id(str(audio_id)))
    session_id = uuid.uuid4().hex
    record = {
        "session_id": session_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "audio_id": str(audio_id),
        "audio_url": f"/api/audio/{audio_id}",
        "locale": body.get("locale") or pipeline.DEFAULT_LOCALE,
        "transcript": body.get("transcript") or "",
        "regional": body.get("regional") or "",
        "english": body.get("english") or "",
        "final": body.get("final") or "",
        "language": body.get("language") or "",
        "regionalUsable": body.get("regionalUsable"),
        "provider": body.get("provider") or "",
        "extraction": body.get("extraction") or None,
        "corrections": body.get("corrections") or {},
    }
    if not audio_path.exists():  # defensive check in case storage changes
        abort(404, "audio not found")
    (SESSION_DIR / f"{session_id}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return jsonify(record)


@app.get("/api/session/<session_id>")
def get_session(session_id):
    path = SESSION_DIR / f"{_safe_id(session_id)}.json"
    if not path.exists():
        abort(404, "session not found")
    return app.response_class(path.read_text(encoding="utf-8"), mimetype="application/json")


@app.post("/api/transcribe")
def transcribe():
    """All-stage transcription. Accepts either an uploaded 'audio' file or a stored
    'audio_id' (form field or JSON). Returns regional / english / final stages."""
    locale = (request.form.get("locale")
              or (request.json.get("locale") if request.is_json else None)
              or pipeline.DEFAULT_LOCALE)

    f = request.files.get("audio")
    if f:
        data, filename, mimetype = f.read(), f.filename, (f.mimetype or "")
    else:
        audio_id = request.form.get("audio_id") or (request.json.get("audio_id") if request.is_json else None)
        if not audio_id:
            abort(400, "provide an 'audio' file or an 'audio_id'")
        path = _find_audio(_safe_id(audio_id))
        data, filename, mimetype = path.read_bytes(), path.name, ""

    try:
        stages = pipeline.transcribe_stages(data, filename, mimetype, locale)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": "transcription failed", "detail": str(e)}), 502
    return jsonify(stages)


@app.post("/api/extract")
def extract():
    """Extract structured medical data from a (final) transcript, store it, and
    return a result_id for download."""
    body = request.get_json(silent=True) or {}
    transcript = body.get("transcript") if isinstance(body.get("transcript"), str) else ""
    if not transcript.strip():
        abort(400, "field 'transcript' (non-empty string) is required")
    locale = body.get("locale") or pipeline.DEFAULT_LOCALE
    try:
        extraction = pipeline.extract(transcript, locale)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": "extraction failed", "detail": str(e)}), 502

    result_id = uuid.uuid4().hex
    record = {
        "result_id": result_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "locale": locale,
        "transcript": transcript,
        "extraction": extraction,
    }
    (RESULT_DIR / f"{result_id}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return jsonify({
        "result_id": result_id,
        "extraction": extraction,
        "download_url": f"/api/result/{result_id}/download",
    })


@app.post("/api/diarize")
def diarize_route():
    """Label a transcript by speaker.

    For 'ai' mode (default): uses Groq to label as Doctor/Patient.
    For 'hybrid' mode: uses frequency + embedding clustering (auto-detects 2 or
    3 speakers) then Groq to relabel, and returns per-speaker transcript
    containers ready for the UI.

    Accepts multipart/form-data or application/json.
    Form fields / JSON keys:
        transcript  — (required) the final merged transcript
        locale      — BCP-47 locale, e.g. "hi_IN"
        mode        — "ai" (default) or "hybrid"
        audio       — audio file (required for hybrid mode)
        max_speakers — 2 or 3 (hybrid only, default 3)
    """
    # Support both JSON and multipart
    if request.is_json:
        body = request.get_json(silent=True) or {}
        transcript = body.get("transcript", "")
        locale = body.get("locale") or pipeline.DEFAULT_LOCALE
        mode = body.get("mode", "ai")
        max_speakers = int(body.get("max_speakers", 3))
        audio_file = None
    else:
        transcript = request.form.get("transcript", "")
        locale = request.form.get("locale") or pipeline.DEFAULT_LOCALE
        mode = request.form.get("mode", "ai")
        max_speakers = int(request.form.get("max_speakers", 3))
        audio_file = request.files.get("audio")

    if not isinstance(transcript, str) or not transcript.strip():
        abort(400, "field 'transcript' (non-empty string) is required")

    if mode == "hybrid":
        if not audio_file:
            abort(400, "hybrid mode requires an 'audio' file")
        file_bytes = audio_file.read()
        filename = audio_file.filename or "audio.webm"
        mimetype = audio_file.mimetype or "audio/webm"
        try:
            result = pipeline.hybrid_diarize(
                file_bytes, filename, mimetype, transcript, locale,
                max_speakers=max(2, min(3, max_speakers)),
            )
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": "hybrid diarization failed", "detail": str(e)}), 502
        return jsonify(result)

    # AI-only mode (original behaviour)
    try:
        return jsonify({"diarized": pipeline.diarize(transcript, locale)})
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": "diarization failed", "detail": str(e)}), 502


@app.post("/api/diarize/retranscribe")
def diarize_retranscribe_route():
    """Re-run transcription for one speaker from a prior hybrid-diarize result.

    Multipart form fields:
        audio    — (required) the original recording
        segments — (required) JSON array of {speaker, start, end}, from the
                   'segments' field of the original /api/diarize hybrid response
        speaker  — (required) which speaker label (A/B/C) to retranscribe
        locale   — BCP-47 locale, e.g. "hi_IN"
    """
    audio_file = request.files.get("audio")
    if not audio_file:
        abort(400, "field 'audio' (file) is required")
    segments_raw = request.form.get("segments", "")
    speaker = request.form.get("speaker", "")
    if not segments_raw or not speaker:
        abort(400, "fields 'segments' (JSON array) and 'speaker' are required")
    try:
        segments = json.loads(segments_raw)
    except ValueError:
        abort(400, "'segments' must be valid JSON")
    locale = request.form.get("locale") or pipeline.DEFAULT_LOCALE

    file_bytes = audio_file.read()
    filename = audio_file.filename or "audio.webm"
    mimetype = audio_file.mimetype or "audio/webm"
    try:
        result = pipeline.retranscribe_speaker(file_bytes, filename, mimetype, segments, speaker, locale)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": "retranscription failed", "detail": str(e)}), 502
    return jsonify(result)


@app.get("/api/result/<result_id>")
def get_result(result_id):
    path = RESULT_DIR / f"{_safe_id(result_id)}.json"
    if not path.exists():
        abort(404, "result not found")
    return app.response_class(path.read_text(encoding="utf-8"), mimetype="application/json")


@app.get("/api/result/<result_id>/download")
def download_result(result_id):
    path = RESULT_DIR / f"{_safe_id(result_id)}.json"
    if not path.exists():
        abort(404, "result not found")
    return send_file(path, as_attachment=True, download_name=f"voicerx-extraction-{result_id}.json")


if __name__ == "__main__":
    import os
    # Dedicated port var so we don't inherit the Node server's PORT from .env.
    app.run(host="0.0.0.0", port=int(os.getenv("FLASK_PORT", "5005")))
