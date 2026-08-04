#!/usr/bin/env python3
"""
IndicConformer STT — OpenAI-compatible /audio/transcriptions server.

Wraps AI4Bharat's IndicConformer-600M (quantized int8 ONNX) via the
`indic-asr-onnx` package. Exposes the same endpoint shape as OpenAI/Groq Whisper
so VoiceRX can use it with TRANSCRIPTION_PROVIDER=custom — no app code changes.

Run (from this dir):
    .venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8001
"""
import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Load the repo-root .env (same file the Node server and other Python
# services read) so HF_TOKEN etc. are available without a separate copy.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# "rnnt" (best accuracy) or "ctc" (faster)
PRECISION = os.getenv("INDIC_PRECISION", "fp32").lower()  # "fp32" (accurate) | "int8" (small)
DECODING = os.getenv("INDIC_DECODING", "rnnt").lower()
DEFAULT_LANG = os.getenv("INDIC_DEFAULT_LANG", "hi").lower()

# 22 languages supported by IndicConformer (ISO-639 codes).
SUPPORTED = {
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "mai", "ml", "mr",
    "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur",
}

app = FastAPI(title="IndicConformer STT", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_transcriber = None


def get_transcriber():
    """Lazily load the model (downloads on first call)."""
    global _transcriber
    if _transcriber is None:
        if PRECISION == "fp32":
            from fp32_transcriber import Fp32IndicTranscriber
            _transcriber = Fp32IndicTranscriber()
        else:
            from indic_asr_onnx import IndicTranscriber
            _transcriber = IndicTranscriber()
    return _transcriber


def to_wav_16k_mono(data: bytes, suffix: str) -> str:
    """Transcode uploaded audio (webm/m4a/ogg…) to 16 kHz mono WAV via ffmpeg."""
    tmp_in = Path(tempfile.gettempdir()) / f"indic_{uuid.uuid4().hex}{suffix}"
    tmp_out = Path(tempfile.gettempdir()) / f"indic_{uuid.uuid4().hex}.wav"
    tmp_in.write_bytes(data)
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(tmp_in), "-ac", "1", "-ar", "16000",
             "-af", "highpass=f=80,lowpass=f=7600,dynaudnorm=f=150:g=15",
             str(tmp_out)],
            check=True,
        )
    finally:
        tmp_in.unlink(missing_ok=True)
    return str(tmp_out)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "precision": PRECISION,
        "decoding": DECODING,
        "default_language": DEFAULT_LANG,
        "model_loaded": _transcriber is not None,
        "supported_languages": sorted(SUPPORTED),
    }


@app.post("/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    # Accepted for OpenAI compatibility; ignored by this model.
    model: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    temperature: str | None = Form(default=None),
    response_format: str | None = Form(default=None),
):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    lang = (language or DEFAULT_LANG)[:2].lower()
    if lang not in SUPPORTED:
        lang = DEFAULT_LANG  # e.g. "en" → fall back (IndicConformer is Indic-only)

    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    wav_path = to_wav_16k_mono(data, suffix)
    try:
        t = get_transcriber()
        text = (
            t.transcribe_ctc(wav_path, lang)
            if DECODING == "ctc"
            else t.transcribe_rnnt(wav_path, lang)
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        Path(wav_path).unlink(missing_ok=True)

    return {
        "text": (text or "").strip(),
        "language": lang,
        "model": f"ai4bharat/indic-conformer-600m ({PRECISION})",
    }


@app.post("/speaker-diarization")
async def speaker_diarization(
    file: UploadFile = File(...),
    transcript: str | None = Form(default=None),
    max_speakers: int = Form(default=3),
):
    """Return speaker segments + per-speaker transcript chunks.

    Optional form fields:
        transcript   — the final merged transcript text; when provided the
                       response includes speakerChunks keyed A/B/C.
        max_speakers — maximum number of speakers to detect (2 or 3, default 3).
    """
    from speaker_diarization import diarize

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    wav_path = to_wav_16k_mono(data, suffix)
    try:
        return diarize(
            wav_path,
            transcript=transcript or "",
            max_speakers=max(2, min(3, max_speakers)),
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Speaker diarization failed: {e}")
    finally:
        Path(wav_path).unlink(missing_ok=True)


@app.post("/speaker-diarization-transcribe")
async def speaker_diarization_transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    max_speakers: int = Form(default=3),
):
    """Diarize audio and independently transcribe each speaker's audio track.

    This is the high-quality endpoint. It:
      1. Runs embedding + pitch-frequency clustering to detect 2–3 speakers.
      2. Carves out each speaker's audio (silence-filling non-speaker regions
         so temporal context is preserved).
      3. Runs the IndicConformer model independently on each speaker track.
      4. Returns segments, frequency stats, and per-speaker transcripts.

    Form fields:
        file         — audio file (any format; ffmpeg will transcode)
        language     — 2-letter ISO-639 code (default: INDIC_DEFAULT_LANG env var)
        max_speakers — 2 or 3 (default 3)

    Response adds to the /speaker-diarization shape:
        speakerTranscripts — {A: "full text…", B: "full text…", C?: "…"}
        speakerChunks      — same but wrapped as single-item lists for UI compat
    """
    from speaker_diarization import diarize_and_transcribe

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    lang = ((language or DEFAULT_LANG)[:2]).lower()
    if lang not in SUPPORTED:
        lang = DEFAULT_LANG

    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    wav_path = to_wav_16k_mono(data, suffix)
    try:
        result = diarize_and_transcribe(
            wav_path,
            lang=lang,
            transcriber=get_transcriber(),
            decoding=DECODING,
            max_speakers=max(2, min(3, max_speakers)),
        )
        return result
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Speaker diarization+transcription failed: {e}")
    finally:
        Path(wav_path).unlink(missing_ok=True)


@app.post("/speaker-diarization-retranscribe")
async def speaker_diarization_retranscribe(
    file: UploadFile = File(...),
    segments: str = Form(...),
    speaker: str = Form(...),
    language: str | None = Form(default=None),
):
    """Re-run transcription for one speaker from an existing diarization result.

    Reuses the original segment boundaries (does not re-run speaker splitting).

    Form fields:
        file     — original audio (any format; ffmpeg will transcode)
        segments — JSON array of {speaker, start, end}, from a prior
                   /speaker-diarization* response for this same recording
        speaker  — which speaker label (A/B/C) to retranscribe
        language — 2-letter ISO-639 code (default: INDIC_DEFAULT_LANG env var)
    """
    from speaker_diarization import retranscribe_speaker

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    try:
        seg_list = json.loads(segments)
    except ValueError:
        raise HTTPException(status_code=400, detail="'segments' must be a JSON array.")

    lang = ((language or DEFAULT_LANG)[:2]).lower()
    if lang not in SUPPORTED:
        lang = DEFAULT_LANG

    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    wav_path = to_wav_16k_mono(data, suffix)
    try:
        text = retranscribe_speaker(
            wav_path, seg_list, speaker, lang, get_transcriber(), decoding=DECODING
        )
        return {"speaker": speaker, "transcript": text}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Retranscription failed: {e}")
    finally:
        Path(wav_path).unlink(missing_ok=True)


@app.post("/speaker-diarization-pyannote")
async def speaker_diarization_pyannote(
    file: UploadFile = File(...),
    max_speakers: int = Form(default=3),
):
    """Same contract as /speaker-diarization, using the pyannote neural
    pipeline instead of resemblyzer+KMeans — real acoustic turn boundaries
    rather than a heuristic distribution of the transcript over them.

    Requires HF_TOKEN (repo-root .env) with the pyannote license accepted;
    see speaker_diarization_pyannote.py's module docstring.
    """
    from speaker_diarization_pyannote import diarize

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    wav_path = to_wav_16k_mono(data, suffix)
    try:
        return diarize(wav_path, max_speakers=max(2, min(3, max_speakers)))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Pyannote speaker diarization failed: {e}")
    finally:
        Path(wav_path).unlink(missing_ok=True)


@app.post("/speaker-diarization-transcribe-pyannote")
async def speaker_diarization_transcribe_pyannote(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    max_speakers: int = Form(default=3),
):
    """Same contract as /speaker-diarization-transcribe, using pyannote for
    the acoustic clustering step, then independently transcribing each
    speaker's carved-out audio exactly as the resemblyzer path does.
    """
    from speaker_diarization_pyannote import diarize_and_transcribe

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    lang = ((language or DEFAULT_LANG)[:2]).lower()
    if lang not in SUPPORTED:
        lang = DEFAULT_LANG

    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    wav_path = to_wav_16k_mono(data, suffix)
    try:
        return diarize_and_transcribe(
            wav_path,
            lang=lang,
            transcriber=get_transcriber(),
            decoding=DECODING,
            max_speakers=max(2, min(3, max_speakers)),
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Pyannote speaker diarization+transcription failed: {e}")
    finally:
        Path(wav_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8001")))
