"""Local speaker embeddings combined with pitch/frequency features.

Auto-detects whether there are 2 or 3 distinct voices using silhouette score,
clusters them, and returns both segment-level results and per-speaker
transcript chunk lists so callers can build individual speaker containers.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

_encoder: VoiceEncoder | None = None

# Labels in A/B/C order so the first detected voice is always "A".
_SPEAKER_LABELS = ["A", "B", "C"]


def _get_encoder() -> VoiceEncoder:
    global _encoder
    if _encoder is None:
        _encoder = VoiceEncoder("cpu")
    return _encoder


def _pitch(samples: np.ndarray, rate: int = 16000) -> float:
    """Estimate fundamental frequency via autocorrelation (YIN-lite)."""
    samples = samples.astype(np.float32)
    if len(samples) < 160 or float(np.sqrt(np.mean(samples * samples))) < 0.01:
        return 0.0
    best_lag, best_corr = 0, 0.0
    for lag in range(53, 201):          # ~80 Hz – 300 Hz range
        left, right = samples[lag:], samples[:-lag]
        denom = float(np.sqrt(np.sum(left * left) * np.sum(right * right))) or 1.0
        corr = float(np.sum(left * right)) / denom
        if corr > best_corr:
            best_corr, best_lag = corr, lag
    return rate / best_lag if best_corr >= 0.30 and best_lag else 0.0


def _build_features(embeddings: np.ndarray, pitch_values: list[float]) -> np.ndarray:
    """Build stable speaker features from voiceprints plus smoothed frequency.

    Pitch is a speaker cue, not an identity by itself: one person can move
    through a large pitch range while speaking.  The voice embedding therefore
    remains the primary feature and the frequency cue is deliberately smaller.
    """
    voiced = [p for p in pitch_values if p > 0]
    mean = float(np.mean(np.log(voiced))) if voiced else 0.0
    std = float(np.std(np.log(voiced))) or 1.0

    features = []
    for embedding, pitch in zip(embeddings, pitch_values):
        embedding = embedding / (float(np.linalg.norm(embedding)) or 1.0)
        frequency_feature = (math.log(pitch) - mean) / std if pitch > 0 else 0.0
        features.append(
            np.concatenate([embedding, np.array([frequency_feature * 0.40], dtype=np.float32)])
        )
    return np.asarray(features, dtype=np.float32)


# A real second speaker's mean pitch differs by tens of Hz, not a handful.
# Calibrated against real recordings: two genuine 2-speaker conversations
# showed 55-64 Hz mean-pitch gaps between clusters; a single excited/tired
# speaker whose silhouette score alone looked like a valid split (0.187,
# just above the 0.16 cutoff below) showed only a 4 Hz gap. Silhouette score
# alone cannot reliably tell these apart — the false split scored close
# behind (0.187) the real ones (0.20-0.24) — so pitch gap is a required
# second check, not just a small feature weight inside the clustering itself.
MIN_PITCH_GAP_HZ = 20.0


def _cluster_pitch_gap(labels: list[int], pitch_values: list[float]) -> float:
    """Largest gap between any two clusters' mean voiced pitch (Hz)."""
    means: dict[int, float] = {}
    for label in set(labels):
        voiced = [p for l, p in zip(labels, pitch_values) if l == label and p > 0]
        if voiced:
            means[label] = float(np.mean(voiced))
    if len(means) < 2:
        return 0.0
    values = list(means.values())
    return max(values) - min(values)


def _pick_n_speakers(features: np.ndarray, pitch_values: list[float], max_speakers: int = 3) -> int:
    """Return 1, 2, or 3 based on cluster quality AND pitch separation.

    A single voice with changing pitch must not be split into multiple people.
    We only accept a split when its silhouette is useful AND the resulting
    clusters' mean pitch differs by more than natural single-speaker
    variation (see MIN_PITCH_GAP_HZ), and require a larger silhouette
    improvement before adding a third speaker.
    """
    if len(features) < 4:
        return 1

    scores: dict[int, float] = {}
    labels_by_k: dict[int, list[int]] = {}
    for k in range(2, max(2, min(3, max_speakers)) + 1):
        if len(features) < k + 1:
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=7).fit(features)
        if len(set(km.labels_)) < k:
            continue
        labels_by_k[k] = km.labels_.tolist()
        try:
            scores[k] = float(silhouette_score(features, km.labels_))
        except Exception:
            scores[k] = -1.0

    # Below this level, a split is more likely to represent pitch movement,
    # room noise, or a breath than a second person.
    if scores.get(2, -1.0) < 0.16:
        return 1
    if _cluster_pitch_gap(labels_by_k[2], pitch_values) < MIN_PITCH_GAP_HZ:
        return 1
    if (3 in scores and scores[3] - scores[2] > 0.08 and scores[3] >= 0.20
            and _cluster_pitch_gap(labels_by_k[3], pitch_values) >= MIN_PITCH_GAP_HZ):
        return 3
    return 2


def _smooth_pitch(values: list[float]) -> list[float]:
    """Median-smooth neighboring pitch windows while preserving pitch changes."""
    if len(values) < 3:
        return values
    smoothed = values[:]
    for index in range(1, len(values) - 1):
        window = [value for value in values[index - 1:index + 2] if value > 0]
        if window:
            smoothed[index] = float(np.median(window))
    return smoothed


def _merge_segments(labels: list[int], starts: list[float], ends: list[float]) -> list[dict]:
    """Collapse adjacent same-speaker windows into contiguous segments."""
    if not labels:
        return []
    result: list[dict] = []
    for label, start, end in zip(labels, starts, ends):
        if result and result[-1]["speaker"] == int(label) and start - result[-1]["end"] < 1.0:
            result[-1]["end"] = end
        else:
            result.append({"speaker": int(label), "start": round(start, 3), "end": round(end, 3)})
    # Drop very short slivers (likely artefacts)
    result = [s for s in result if s["end"] - s["start"] >= 0.25]
    # Resolve overlapping boundaries from partial embedding windows
    for previous, current in zip(result, result[1:]):
        boundary = (previous["end"] + current["start"]) / 2
        previous["end"] = round(boundary, 3)
        current["start"] = round(boundary, 3)
    return result


def _label_segments(raw: list[dict]) -> list[dict]:
    """Replace integer cluster IDs with stable A/B/C labels in first-seen order."""
    seen: dict[int, str] = {}
    labeled = []
    for seg in raw:
        numeric = seg["speaker"]
        if numeric not in seen:
            seen[numeric] = _SPEAKER_LABELS[len(seen)]
        labeled.append({**seg, "speaker": seen[numeric]})
    return labeled


def _label_map(raw: list[dict]) -> dict[int, str]:
    seen: dict[int, str] = {}
    for seg in raw:
        numeric = int(seg["speaker"])
        if numeric not in seen:
            seen[numeric] = _SPEAKER_LABELS[len(seen)]
    return seen


def _frequency_groups(
    labels: list[int], pitch_values: list[float], label_map: dict[int, str]
) -> list[dict]:
    """Summarize the changing frequency range belonging to each voice group."""
    grouped: dict[str, list[float]] = {}
    for label, pitch in zip(labels, pitch_values):
        if pitch > 0 and label in label_map:
            grouped.setdefault(label_map[label], []).append(pitch)

    return [
        {
            "speaker": speaker,
            "minHz": round(float(np.percentile(values, 10)), 1),
            "maxHz": round(float(np.percentile(values, 90)), 1),
            "meanHz": round(float(np.median(values)), 1),
            "samples": len(values),
        }
        for speaker, values in grouped.items()
    ]


def _per_speaker_chunks(segments: list[dict], transcript: str) -> dict[str, list[str]]:
    """Distribute transcript sentences across speakers proportionally by segment time.

    This is a best-effort distribution: we split the transcript into sentences,
    estimate each sentence's midpoint in time (proportional to character count),
    and assign it to the speaker active at that time. Sentences with no active
    segment keep the last known speaker.
    """
    if not segments or not transcript.strip():
        return {}

    # Split transcript into sentences
    import re
    sentences = [s.strip() for s in re.split(r"(?<=[.!?।\n])\s*", transcript) if s.strip()]
    if not sentences:
        return {}

    # STT often returns a paragraph with no punctuation. In that case, assigning
    # the whole paragraph by its midpoint incorrectly attributes every word to
    # one speaker. Distribute words across the acoustic segments instead.
    if len(sentences) == 1 and len(segments) > 1:
        tokens = transcript.split()
        duration = max(s["end"] for s in segments)
        total_chars = sum(len(token) for token in tokens) or 1
        chunks: dict[str, list[str]] = {}
        current_speaker = ""
        current_words: list[str] = []
        char_offset = 0

        def flush() -> None:
            nonlocal current_words
            if current_speaker and current_words:
                chunks.setdefault(current_speaker, []).append(" ".join(current_words))
                current_words = []

        seg_idx = 0
        for token in tokens:
            midpoint = ((char_offset + len(token) / 2) / total_chars) * duration
            while seg_idx + 1 < len(segments) and midpoint > segments[seg_idx]["end"]:
                seg_idx += 1
            speaker = segments[seg_idx]["speaker"]
            if speaker != current_speaker:
                flush()
                current_speaker = speaker
            current_words.append(token)
            char_offset += len(token) + 1
        flush()
        return chunks

    duration = max(s["end"] for s in segments)
    total_chars = sum(len(s) for s in sentences) or 1

    chunks: dict[str, list[str]] = {}
    seg_idx = 0
    char_offset = 0

    for sentence in sentences:
        # Estimate midpoint in audio time for this sentence
        midpoint = ((char_offset + len(sentence) / 2) / total_chars) * duration
        # Advance segment pointer
        while seg_idx + 1 < len(segments) and midpoint > segments[seg_idx]["end"]:
            seg_idx += 1
        speaker = segments[seg_idx]["speaker"]
        chunks.setdefault(speaker, []).append(sentence)
        char_offset += len(sentence)

    return chunks


def _attach_transcript(segments: list[dict], transcript: str) -> list[dict]:
    """Attach best-effort sentence text to each acoustic segment.

    The current STT response has no word timestamps, so sentence midpoints are
    mapped proportionally across the detected recording duration.
    """
    if not segments or not transcript.strip():
        return segments

    duration = max(float(segment["end"]) for segment in segments)
    attached = []
    for index, segment in enumerate(segments):
        start = float(segment["start"])
        end = float(segment["end"])
        start_index = round((start / duration) * len(transcript)) if duration else 0
        end_index = round((end / duration) * len(transcript)) if duration else len(transcript)
        # Keep adjacent chunks readable by expanding boundaries to whitespace.
        if index > 0:
            while start_index < len(transcript) and not transcript[start_index].isspace():
                start_index += 1
        if index < len(segments) - 1:
            while end_index > start_index and not transcript[end_index - 1].isspace():
                end_index -= 1
        text = transcript[start_index:end_index].strip()
        attached.append({**segment, "text": text})
    return attached


# ── Per-speaker audio extraction ──────────────────────────────────────────────

def extract_speaker_audio(
    wav: np.ndarray,
    segments: list[dict],
    rate: int = 16000,
    silence_fill: bool = True,
) -> dict[str, np.ndarray]:
    """Carve out each speaker's audio from the full WAV.

    For each speaker we build a new array the same length as the full recording.
    Regions that belong to this speaker are copied verbatim; all other regions
    are either zeroed out (silence_fill=True, preserves time alignment) or
    concatenated tightly (silence_fill=False, shorter but loses alignment).

    Args:
        wav           — mono float32 PCM array at `rate` Hz
        segments      — list of {speaker, start, end} from diarize()
        rate          — sample rate (must match wav)
        silence_fill  — when True, keep a silence gap for other speakers' turns
                        so the extracted track stays time-aligned to the original

    Returns:
        dict mapping speaker label ("A", "B", "C") → float32 numpy array
    """
    total = len(wav)
    speakers = sorted({s["speaker"] for s in segments})

    # Build a sample-level speaker map — who is speaking at each sample?
    # Default to first speaker so silence regions don't break the array.
    owner = np.full(total, -1, dtype=np.int8)   # -1 = unassigned / silence
    for seg in segments:
        start_s = max(0, int(round(seg["start"] * rate)))
        end_s   = min(total, int(round(seg["end"] * rate)))
        spk_idx = speakers.index(seg["speaker"])
        owner[start_s:end_s] = spk_idx

    result: dict[str, np.ndarray] = {}
    for spk_idx, label in enumerate(speakers):
        mask = owner == spk_idx
        if silence_fill:
            # Keep original length; silence out everyone else
            track = wav.copy()
            track[~mask] = 0.0
        else:
            # Only the frames where this speaker is active
            track = wav[mask]
        # Skip a speaker whose total audio is less than 0.25 s
        if len(track) < int(0.25 * rate):
            continue
        result[label] = track.astype(np.float32)

    return result


def transcribe_speaker_audio(
    speaker_wav: dict[str, np.ndarray],
    lang: str,
    transcriber,
    decoding: str = "ctc",
    rate: int = 16000,
) -> dict[str, str]:
    """Run the Indic transcriber on each speaker's extracted audio.

    Writes each speaker's track to a temp WAV file, runs CTC/RNNT decoding,
    and returns a dict of {speaker_label: transcript_text}.

    Args:
        speaker_wav  — output of extract_speaker_audio()
        lang         — 2-letter ISO language code for IndicConformer
        transcriber  — Fp32IndicTranscriber or IndicTranscriber instance
        decoding     — "ctc" or "rnnt"
        rate         — sample rate (must match the extracted audio)
    """
    import tempfile
    import uuid

    import soundfile as sf

    transcripts: dict[str, str] = {}

    for label, audio in speaker_wav.items():
        if len(audio) < int(0.25 * rate):
            transcripts[label] = ""
            continue
        tmp = Path(tempfile.gettempdir()) / f"spk_{label}_{uuid.uuid4().hex}.wav"
        try:
            sf.write(str(tmp), audio, rate, subtype="PCM_16")
            text = (
                transcriber.transcribe_ctc(str(tmp), lang)
                if decoding == "ctc"
                else transcriber.transcribe_rnnt(str(tmp), lang)
            )
            transcripts[label] = (text or "").strip()
        except Exception:
            transcripts[label] = ""
        finally:
            tmp.unlink(missing_ok=True)

    return transcripts


def diarize(audio_path: str, transcript: str = "", max_speakers: int = 3) -> dict:
    """Run frequency + embedding diarization.

    Returns:
        segments       — list of {speaker, start, end} dicts (A/B/C labels)
        detectedVoices — 1, 2, or 3
        frequencyUsed  — always True
        frequencyGroups — per-speaker Hz statistics
        speakerChunks  — {A: [...sentences], B: [...sentences], C?: [...sentences]}
                         populated when a transcript is provided (heuristic distribution)
    """
    wav = preprocess_wav(audio_path)
    encoder = _get_encoder()

    _, embeddings, wav_slices = encoder.embed_utterance(wav, return_partials=True, rate=1.0)

    if len(embeddings) < 3:
        return {
            "segments": [],
            "detectedVoices": 1 if len(wav) >= 1600 else 0,
            "frequencyUsed": True,
            "frequencyGroups": [],
            "speakerChunks": {},
        }

    pitch_values: list[float] = []
    for wav_slice in wav_slices:
        pitch_values.append(_pitch(wav[wav_slice]))
    pitch_values = _smooth_pitch(pitch_values)

    features = _build_features(embeddings, pitch_values)
    n_speakers = _pick_n_speakers(features, pitch_values, max_speakers=max_speakers)

    model = KMeans(n_clusters=n_speakers, n_init=10, random_state=7).fit(features)
    labels = model.labels_.tolist()

    if n_speakers == 1 or len(set(labels)) < 2:
        duration = round(len(wav) / 16000, 3)
        groups = _frequency_groups(labels, pitch_values, {0: "A"})
        segs = [{"speaker": "A", "start": 0.0, "end": duration}] if duration else []
        segs = _attach_transcript(segs, transcript)
        return {
            "segments": segs,
            "detectedVoices": 1,
            "frequencyUsed": True,
            "frequencyGroups": groups,
            "speakerChunks": _per_speaker_chunks(segs, transcript) if transcript.strip() and duration else {},
        }

    starts = [s.start / 16000 for s in wav_slices]
    ends   = [s.stop  / 16000 for s in wav_slices]
    raw = _merge_segments(labels, starts, ends)

    actual_voices = len({s["speaker"] for s in raw})
    label_map = _label_map(raw)
    segments = _attach_transcript(_label_segments(raw), transcript)

    if actual_voices < 2:
        return {
            "segments": segments,
            "detectedVoices": 1,
            "frequencyUsed": True,
            "frequencyGroups": _frequency_groups(labels, pitch_values, label_map),
            "speakerChunks": _per_speaker_chunks(segments, transcript) if transcript.strip() else {},
        }

    chunks = _per_speaker_chunks(segments, transcript) if transcript.strip() else {}

    return {
        "segments": segments,
        "detectedVoices": actual_voices,
        "frequencyUsed": True,
        "frequencyGroups": _frequency_groups(labels, pitch_values, label_map),
        "speakerChunks": chunks,
    }


def diarize_and_transcribe(
    audio_path: str,
    lang: str,
    transcriber,
    decoding: str = "ctc",
    max_speakers: int = 3,
) -> dict:
    """Diarize audio then independently transcribe each speaker's audio.

    This is the high-quality path: instead of distributing a single joint
    transcript by heuristic, we:
      1. Run diarize() to get speaker segments.
      2. Extract each speaker's audio using silence-filled carve-outs so the
         model sees correct temporal context.
      3. Run the Indic transcriber independently on each speaker track.
      4. Return everything: segments, frequency groups, per-speaker audio
         transcripts, and the heuristic chunks as a fallback.

    Args:
        audio_path  — path to preprocessed 16 kHz mono WAV
        lang        — 2-letter ISO language code
        transcriber — Fp32IndicTranscriber or IndicTranscriber instance
        decoding    — "ctc" or "rnnt"
        max_speakers — maximum speakers to detect (2 or 3)

    Returns dict with all keys from diarize() plus:
        speakerTranscripts — {A: "full text…", B: "full text…", C?: "…"}
                             independently transcribed per-speaker text
    """
    base = diarize(audio_path, transcript="", max_speakers=max_speakers)
    segments = base.get("segments", [])
    detected = base.get("detectedVoices", 0)

    if detected < 2 or not segments:
        # Single speaker or no segments — transcribe the whole file as speaker A
        wav = preprocess_wav(audio_path)
        speaker_wav = {"A": wav}
        transcripts = transcribe_speaker_audio(speaker_wav, lang, transcriber, decoding)
        base["speakerTranscripts"] = transcripts
        return base

    # Extract per-speaker audio (silence-filled, preserves timing context)
    wav = preprocess_wav(audio_path)
    speaker_wav = extract_speaker_audio(wav, segments, rate=16000, silence_fill=True)

    # Independent transcription per speaker
    speaker_transcripts = transcribe_speaker_audio(speaker_wav, lang, transcriber, decoding)

    base["speakerTranscripts"] = speaker_transcripts
    # Also populate speakerChunks from the independent transcripts
    # (each transcript is already the speaker's full text; wrap in a list)
    base["speakerChunks"] = {
        spk: [txt] for spk, txt in speaker_transcripts.items() if txt.strip()
    }
    return base


def retranscribe_speaker(
    audio_path: str,
    segments: list[dict],
    speaker: str,
    lang: str,
    transcriber,
    decoding: str = "ctc",
) -> str:
    """Re-run transcription for a single already-diarized speaker.

    Reuses the original segment boundaries as-is (the acoustic speaker split
    is not re-run) so a caller can retry just one speaker's transcript — e.g.
    after a garbled first pass — without paying for full diarization again.
    """
    wav = preprocess_wav(audio_path)
    speaker_wav = extract_speaker_audio(wav, segments, rate=16000, silence_fill=True)
    track = speaker_wav.get(speaker)
    if track is None:
        return ""
    transcripts = transcribe_speaker_audio({speaker: track}, lang, transcriber, decoding)
    return transcripts.get(speaker, "")
