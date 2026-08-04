"""Acoustic speaker diarization via pyannote's neural pipeline.

An alternative engine to speaker_diarization.py's resemblyzer+KMeans approach.
Produces the same {segments, detectedVoices, frequencyUsed, frequencyGroups,
speakerChunks} shape, so server.py's pyannote-backed endpoints can reuse the
existing per-speaker audio extraction and independent transcription helpers
unchanged.

Requires HF_TOKEN (see repo-root .env) and, once, accepting the license for:
  - https://huggingface.co/pyannote/speaker-diarization-3.1
  - https://huggingface.co/pyannote/segmentation-3.0
  - https://huggingface.co/pyannote/speaker-diarization-community-1
"""
from __future__ import annotations

import os

_pipeline = None

# Labels in A/B/C order so the first detected voice is always "A" — matches
# speaker_diarization.py's convention so downstream code doesn't care which
# engine produced the segments.
_SPEAKER_LABELS = ["A", "B", "C"]


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        import torch
        from pyannote.audio import Pipeline

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN is not set. Pyannote diarization needs a free Hugging Face "
                "token, with the license accepted for pyannote/speaker-diarization-3.1, "
                "pyannote/segmentation-3.0, and pyannote/speaker-diarization-community-1."
            )
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        pipeline.to(torch.device(device))
        _pipeline = pipeline
    return _pipeline


def _merge_segments(raw: list[dict], gap: float = 1.2, min_duration: float = 0.3) -> list[dict]:
    """Collapse same-speaker turns separated by a short pause into one turn,
    and drop slivers shorter than min_duration.

    Pyannote's raw output is frequently over-segmented — brief interjections,
    breaths, or a single fragmented word can appear as their own turn. Left
    unmerged, each one would become a separate (slow, costly) transcription
    call downstream. This mirrors speaker_diarization.py's own merge/filter
    step so both engines behave consistently.
    """
    if not raw:
        return []
    merged: list[dict] = []
    for seg in raw:
        if merged and merged[-1]["speaker"] == seg["speaker"] and seg["start"] - merged[-1]["end"] < gap:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    return [seg for seg in merged if seg["end"] - seg["start"] >= min_duration]


def diarize(audio_path: str, transcript: str = "", max_speakers: int = 3) -> dict:
    """Run pyannote diarization. Same return shape as speaker_diarization.diarize()."""
    pipeline = _get_pipeline()
    result = pipeline(audio_path, max_speakers=max(2, min(3, max_speakers)))
    # pyannote.audio 4.x wraps the Annotation in a DiarizeOutput dataclass;
    # 3.x returns the Annotation directly. Support both.
    annotation = getattr(result, "speaker_diarization", result)

    raw = sorted(
        (
            {"speaker": speaker, "start": round(turn.start, 3), "end": round(turn.end, 3)}
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ),
        key=lambda seg: seg["start"],
    )
    raw = _merge_segments(raw)

    # Relabel pyannote's own SPEAKER_00/SPEAKER_01/... into A/B/C in
    # first-appearance order.
    seen: dict[str, str] = {}
    labeled = []
    for seg in raw:
        original = seg["speaker"]
        if original not in seen:
            if len(seen) >= len(_SPEAKER_LABELS):
                continue  # more distinct voices than max_speakers allows; drop the rest
            seen[original] = _SPEAKER_LABELS[len(seen)]
        labeled.append({**seg, "speaker": seen[original]})

    return {
        "segments": labeled,
        "detectedVoices": len(seen),
        "frequencyUsed": False,
        "frequencyGroups": [],
        "speakerChunks": {},
    }


def diarize_and_transcribe(
    audio_path: str,
    lang: str,
    transcriber,
    decoding: str = "ctc",
    max_speakers: int = 3,
) -> dict:
    """Diarize with pyannote, then independently transcribe each speaker's audio.

    Mirrors speaker_diarization.diarize_and_transcribe() but swaps in the
    pyannote engine for the clustering step; per-speaker audio extraction and
    transcription are the same engine-agnostic helpers.
    """
    from resemblyzer import preprocess_wav

    from speaker_diarization import extract_speaker_audio, transcribe_speaker_audio

    base = diarize(audio_path, transcript="", max_speakers=max_speakers)
    segments = base.get("segments", [])
    detected = base.get("detectedVoices", 0)

    wav = preprocess_wav(audio_path)
    speaker_wav = (
        extract_speaker_audio(wav, segments, rate=16000, silence_fill=True)
        if detected >= 2 and segments
        else {"A": wav}
    )

    transcripts = transcribe_speaker_audio(speaker_wav, lang, transcriber, decoding)
    base["speakerTranscripts"] = transcripts
    base["speakerChunks"] = {spk: [txt] for spk, txt in transcripts.items() if txt.strip()}
    return base
