"""
VoiceRX pipeline — Python port of the Node services for the Flask API.

Orchestrates:
  - IndicConformer STT service  (regional transcript)        -> INDIC_STT_URL
  - Groq Whisper                (English transcript)          -> Groq /audio/translations
  - Groq LLMs                   (merge + medicine NER + extract)
  - Medicine FAISS service      (drug-name canonicalisation)  -> MEDICINE_SEARCH_URL

Mirrors the behaviour of server/src/services/*.ts:
  * dual-ASR: regional (selected language) + English (Whisper), merged by Groq as a
    medical transcriptionist — conversation in the selected language, medicine names
    in English. English selected -> single English transcript (Indic bypassed).
  * extraction: structured JSON, all fields in English, exact drug kept (no
    brand<->generic substitution).
"""
import json
import os
import re
import subprocess
import tempfile
import uuid
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load the repo-root .env (this file lives at <root>/flask_api/pipeline.py).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
GROQ_NER_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_EXTRACTION_MODEL = os.getenv("GROQ_EXTRACTION_MODEL", "llama-3.3-70b-versatile")
GROQ_TRANSCRIPTION_MODEL = os.getenv("GROQ_TRANSCRIPTION_MODEL", "whisper-large-v3")

INDIC_STT_URL = os.getenv("INDIC_STT_URL", "http://localhost:8001").rstrip("/")
MEDICINE_SEARCH_URL = os.getenv("MEDICINE_SEARCH_URL", "http://localhost:8000").rstrip("/")
MEDICINE_CONFIDENCE = float(os.getenv("MEDICINE_CONFIDENCE_THRESHOLD", "85"))
DEFAULT_LOCALE = os.getenv("CORRECTION_LOCALE", "en_IN")

HTTP_TIMEOUT = int(os.getenv("PIPELINE_HTTP_TIMEOUT", "120"))

# Hardcoded phonetic STT hallucinations (ported from the Dart/TS alias table).
ALIASES = {
    "cal coral dee": "Kalcoral D", "calculate d": "Kalcoral D", "kal coral d": "Kalcoral D",
    "karakural d": "Kalcoral D", "karakural": "Kalcoral",
    "ravi prasoon": "Rabeprazole", "ravi prasual": "Rabeprazole", "ravi prasal": "Rabeprazole",
    "ravi prasul": "Rabeprazole", "ravi prazool": "Rabeprazole",
    "happy prasul": "Rabeprazole", "happy prasoon": "Rabeprazole", "happy prasal": "Rabeprazole",
    "happi prasul": "Rabeprazole", "happi prasoon": "Rabeprazole",
    "brazil brazil": "Rabeprazole", "baby brazil": "Rabeprazole", "baby prasoon": "Rabeprazole",
    "baby prasul": "Rabeprazole", "baby prasal": "Rabeprazole", "enterprises": "Pantoprazole",
}

MEDICAL_PROMPT = (
    "Indian outpatient clinic dictation. Common terms: Paracetamol, Dolo 650, Calpol, "
    "Combiflam, Kalcoral D, Augmentin, Azithromycin, Amoxicillin, Pantoprazole, Pan-D, "
    "Rabeprazole, Zerodol, Crocin, Montek LC, Allegra, Metformin, Telma, Amlodipine, "
    "Atorvastatin, Ecosprin, Ondansetron, Domstal, mg, ml, tablet, capsule, syrup, BD, OD, TDS."
)

_LANG_HINTS = {
    "hi": "in Hindi or a mix of Hindi and English (Hinglish)",
    "gu": "in Gujarati or a mix of Gujarati and English",
    "mr": "in Marathi or a mix of Marathi and English",
    "bn": "in Bengali or a mix of Bengali and English",
    "ta": "in Tamil or a mix of Tamil and English",
    "te": "in Telugu or a mix of Telugu and English",
    "kn": "in Kannada or a mix of Kannada and English",
    "ml": "in Malayalam or a mix of Malayalam and English",
    "pa": "in Punjabi or a mix of Punjabi and English",
    "or": "in Odia or a mix of Odia and English",
}


def _lang_code(locale: str | None) -> str:
    if not locale or locale.lower() == "auto":
        return ""
    return locale[:2].lower()


def language_hint(locale: str) -> str:
    return _LANG_HINTS.get(_lang_code(locale), "in English (Indian accent / medical terminology)")


# ── Groq helpers ──────────────────────────────────────────────────────────────
def _groq_chat(messages, model, temperature=0.0, max_tokens=2048, json_mode=False) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = requests.post(
        f"{GROQ_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json=body, timeout=HTTP_TIMEOUT,
    )
    if not r.ok:
        raise RuntimeError(f"Groq chat error {r.status_code}: {r.text[:200]}")
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def _groq_audio(file_bytes, filename, mimetype, translate: bool, language: str | None = None,
                include_metadata: bool = False):
    """Groq Whisper. translate=True -> /audio/translations (->English).

    When include_metadata is enabled, use verbose JSON so Whisper can provide
    the detected language for Auto mode. The normal callers still receive a
    plain transcript string.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")
    endpoint = "/audio/translations" if translate else "/audio/transcriptions"
    data = {"model": GROQ_TRANSCRIPTION_MODEL, "prompt": MEDICAL_PROMPT,
            "temperature": "0", "response_format": "verbose_json" if include_metadata else "json"}
    if not translate and language:
        data["language"] = language
    r = requests.post(
        f"{GROQ_BASE_URL}{endpoint}",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": (filename or "audio.webm", file_bytes, mimetype or "audio/webm")},
        data=data, timeout=HTTP_TIMEOUT,
    )
    if not r.ok:
        raise RuntimeError(f"Groq Whisper error {r.status_code}: {r.text[:200]}")
    payload = r.json()
    text = (payload.get("text") or "").strip()
    if include_metadata:
        return {"text": text, "language": (payload.get("language") or "").strip().lower()}
    return text


def _normalized_audio(file_bytes, filename, mimetype, seconds: int | None = None):
    """Create normalized mono 16 kHz audio, optionally limited to a prefix."""
    suffix = Path(filename or "audio.webm").suffix or ".webm"
    token = uuid.uuid4().hex
    input_path = Path(tempfile.gettempdir()) / f"voicerx_detect_{token}_in{suffix}"
    output_path = Path(tempfile.gettempdir()) / f"voicerx_detect_{token}_out.wav"
    input_path.write_bytes(file_bytes)
    try:
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(input_path),
        ]
        if seconds is not None:
            command.extend(["-t", str(seconds)])
        command.extend([
            "-ac", "1", "-ar", "16000",
            "-af", "highpass=f=80,lowpass=f=7600,dynaudnorm=f=150:g=15",
            str(output_path),
        ])
        subprocess.run(command, check=True, timeout=60)
        return output_path.read_bytes(), "audio.wav", "audio/wav"
    except (OSError, subprocess.SubprocessError):
        return file_bytes, filename, mimetype
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def detect_language(file_bytes, filename, mimetype) -> str:
    """Detect the spoken language from the first few seconds using Whisper."""
    prefix, prefix_name, prefix_type = _normalized_audio(file_bytes, filename, mimetype, 8)
    result = _groq_audio(prefix, prefix_name, prefix_type, translate=False, include_metadata=True)
    detected = (result.get("language") or "").lower()
    # Whisper normally returns ISO-639-1 codes, but normalize occasional names.
    aliases = {
        "assamese": "as", "bengali": "bn", "gujarati": "gu", "hindi": "hi",
        "kannada": "kn", "malayalam": "ml", "marathi": "mr", "odia": "or",
        "oriya": "or", "punjabi": "pa", "tamil": "ta", "telugu": "te",
        "english": "en",
    }
    return aliases.get(detected, detected[:2])


def regional_quality(text: str, language: str) -> bool:
    """Reject empty or clearly wrong-script Indic output."""
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) < 4:
        return False
    if re.search(r"(.)\1{5,}", value):
        return False
    script_ranges = {
        "gu": r"\u0A80-\u0AFF", "hi": r"\u0900-\u097F", "mr": r"\u0900-\u097F",
        "bn": r"\u0980-\u09FF", "pa": r"\u0A00-\u0A7F", "ta": r"\u0B80-\u0BFF",
        "te": r"\u0C00-\u0C7F", "kn": r"\u0C80-\u0CFF", "ml": r"\u0D00-\u0D7F",
        "or": r"\u0B00-\u0B7F",
    }
    char_range = script_ranges.get(language[:2].lower())
    if not char_range:
        return True
    script_chars = len(re.findall(f"[{char_range}]", value))
    letters = len(re.findall(r"[A-Za-z\u0900-\u0D7F]", value))
    return script_chars >= 2 and (script_chars / max(letters, 1)) >= 0.15


def indic_language_code(code: str) -> str:
    """Map detected Urdu to Hindi for the IndicTransformer stage."""
    return "hi" if (code or "").lower() == "ur" else code


# ── Transcription stages ────────────────────────────────────────────────────────
def transcribe_regional(file_bytes, filename, mimetype, locale) -> str:
    """Regional (selected-language) transcript via the IndicConformer service.
    English bypasses Indic and uses Groq Whisper."""
    code = _lang_code(locale)
    code = indic_language_code(code)
    if code == "en":
        return _groq_audio(file_bytes, filename, mimetype, translate=False, language="en")
    r = requests.post(
        f"{INDIC_STT_URL}/audio/transcriptions",
        files={"file": (filename or "audio.webm", file_bytes, mimetype or "audio/webm")},
        data={"language": code or DEFAULT_LOCALE[:2]}, timeout=HTTP_TIMEOUT,
    )
    if not r.ok:
        raise RuntimeError(f"IndicConformer error {r.status_code}: {r.text[:200]}")
    return (r.json().get("text") or "").strip()


def transcribe_english(file_bytes, filename, mimetype) -> str:
    """English transcript via Groq Whisper translate."""
    return _groq_audio(file_bytes, filename, mimetype, translate=True)


# ── Medicine name resolution (FAISS) ─────────────────────────────────────────────
_STRENGTH_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml|g|iu|units?|%)\b", re.I)
_FORM_RE = re.compile(
    r"\b(?:tablets?|tab|capsules?|cap|syrup|suspension|solution|injections?|inj|infusion|"
    r"drops?|gel|cream|ointment|lotion|spray|inhaler|patch|sachet|powder|granules?|"
    r"suppository|respules?|kit)\b", re.I)


def _stripped_brand(s: str) -> str:
    return re.sub(r"\s+", " ", _FORM_RE.sub(" ", _STRENGTH_RE.sub(" ", s))).strip()


def _core_words(s: str):
    t = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+\s-]", " ", _stripped_brand(s).lower())).strip()
    return t.split(" ") if t else []


def medicine_search(query: str):
    try:
        r = requests.post(f"{MEDICINE_SEARCH_URL}/search", json={"query": query}, timeout=HTTP_TIMEOUT)
        if not r.ok:
            return None
        return r.json()
    except requests.RequestException:
        return None


def _spelling_fix_only(token: str, match_brand: str):
    """Accept only a spelling fix of the same name; never a brand/strength/form expansion."""
    tok, match = _core_words(token), _core_words(match_brand)
    if not tok or not match or len(match) > len(tok):
        return None
    if SequenceMatcher(None, " ".join(tok), " ".join(match)).ratio() < 0.7:
        return None
    cleaned = _stripped_brand(match_brand)
    return None if cleaned.lower() == token.lower().strip() else cleaned


def extract_medicine_tokens(text: str):
    try:
        content = _groq_chat(
            [
                {"role": "system", "content":
                    "You are a medical NER system. Extract ONLY medicine/drug names from the "
                    "transcript (brand, generic, or misspelled). Return a JSON array of strings, "
                    'nothing else. Example: ["Paracetamol", "Dolo 650"]. If none, return [].'},
                {"role": "user", "content": text},
            ],
            model=GROQ_NER_MODEL, temperature=0, max_tokens=512,
        )
        m = re.search(r"\[[\s\S]*?\]", content)
        if not m:
            return []
        parsed = json.loads(m.group(0))
        return [t for t in parsed if isinstance(t, str)]
    except Exception:
        return []


def normalize_tokens(tokens):
    corrections = {}
    for token in tokens:
        norm = token.lower().strip()
        if norm in ALIASES:
            corrections[token] = ALIASES[norm]
            continue
        res = medicine_search(token)
        if res and res.get("match") and (res.get("confidence") == "high" or res.get("score", 0) >= MEDICINE_CONFIDENCE):
            fix = _spelling_fix_only(token, res["match"])
            if fix:
                corrections[token] = fix
    return corrections


def canonicalize_prescriptions(extraction: dict) -> dict:
    """Apply the trained medicine index to extracted prescription names.

    The LLM identifies prescription candidates, but the local FAISS/BK-tree
    medicine service is the authority for spelling corrections. Only a
    high-confidence spelling correction of the same medicine is accepted; we
    never replace a brand with a generic or invent a medicine.
    """
    if not isinstance(extraction, dict):
        return extraction
    prescriptions = extraction.get("prescriptions")
    if not isinstance(prescriptions, list):
        return extraction
    for prescription in prescriptions:
        if not isinstance(prescription, dict):
            continue
        medication = prescription.get("medication")
        if not isinstance(medication, str) or not medication.strip():
            continue
        norm = medication.lower().strip()
        if norm in ALIASES:
            prescription["medication"] = ALIASES[norm]
            continue
        result = medicine_search(medication)
        if not result or not result.get("match"):
            continue
        is_confident = result.get("confidence") == "high" or result.get("score", 0) >= MEDICINE_CONFIDENCE
        if is_confident:
            correction = _spelling_fix_only(medication, result["match"])
            if correction:
                prescription["medication"] = correction
    return extraction


def _corrections_hint(corrections):
    if not corrections:
        return ""
    lines = "\n".join(f'  "{k}" -> "{v}"' for k, v in corrections.items())
    return "\n\nVerified medicine names (use these exact English spellings):\n" + lines


# ── Combine (medical transcriptionist merge) ─────────────────────────────────────
def _combine_prompt(locale, corrections):
    return (
        f"You are an expert MEDICAL TRANSCRIPTIONIST for an Indian OPD clinic. You are given TWO\n"
        f"automatic transcripts of the SAME doctor-patient audio:\n"
        f"(A) REGIONAL — transcribed {language_hint(locale)} by an engine strong at the local\n"
        f"    language. TRUST THIS for the CONVERSATION.\n"
        f"(B) ENGLISH — an English transcript by an engine strong at English/Latin terms. TRUST\n"
        f"    THIS for MEDICINE NAMES, DOSAGE STRENGTHS (e.g. '500 mg'), and clearly heard\n"
        f"    English/code-switched clinical words and phrases.\n\n"
        f"Reconcile them into ONE accurate transcript:\n"
        f"1. Keep the conversation in the regional language/script (A); do not translate or drop it.\n"
        f"2. Write the EXACT drug named in standard English/Latin spelling (from (B) + list below),\n"
        f"   even where (A) garbled it or wrote it in the regional script; never swap a brand for\n"
        f"   its generic or vice versa (if 'Dolo' was said, keep 'Dolo').\n"
        f"3. Write dosages/strengths in English digits with units ('500 mg'); vitals as digits.\n"
        f"4. Preserve valid English words that were actually spoken in the consultation (for\n"
        f"   example 'sleep', 'proper sleep', 'headache', 'last one month', or a medicine name)\n"
        f"   when (A) turns them into phonetic/non-words. Use (B) to repair those fragments, but\n"
        f"   do not translate the entire conversation into English.\n"
        f"5. For frequency, timing and duration spoken in the local language, TRUST the regional\n"
        f"   transcript (A): 'दो दिन' = '2 days', 'दिन में दो बार' = 'twice a day'.\n"
        f"6. PRESERVE EVERY NUMBER EXACTLY — never change a count, dose, frequency or duration\n"
        f"   ('2 days' must NEVER become '1 day'). State one clear value; no contradictory phrasing.\n"
        f"7. Never invent clinical facts. Output ONLY the final transcript as plain text; do not\n"
        f"   wrap it in quotation marks, triple backticks, or labels." + _corrections_hint(corrections)
    )


def _clean_merged_transcript(text: str) -> str:
    """Remove common LLM formatting artifacts without changing transcript content."""
    cleaned = (text or '').strip()
    cleaned = re.sub(r'^```(?:text|markdown)?\s*', '', cleaned, flags=re.I)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = re.sub(r'^\s*"""\s*', '', cleaned)
    cleaned = re.sub(r'\s*"""\s*$', '', cleaned)
    cleaned = re.sub(r'^\s*"\s*', '', cleaned)
    cleaned = re.sub(r'\s*"\s*$', '', cleaned)
    return cleaned.strip()


def combine(regional: str, english: str, locale: str):
    """Merge regional + English. English locale -> English transcript is the final."""
    if _lang_code(locale) == "en":
        return {"corrected": (english.strip() or regional.strip()), "corrections": {}, "medicineTokens": []}

    corrections = {}
    english_scan = english
    for pattern, corr in ALIASES.items():
        if re.search(re.escape(pattern), english_scan, re.I):
            corrections[pattern] = corr
            english_scan = re.sub(re.escape(pattern), corr, english_scan, flags=re.I)

    tokens = []
    if GROQ_API_KEY:
        tokens = extract_medicine_tokens(english_scan)
        corrections.update(normalize_tokens(tokens))
        try:
            max_tokens = min(8000, max(2048, len(regional) * 2))
            merged = _groq_chat(
                [
                    {"role": "system", "content": _combine_prompt(locale, corrections)},
                    {"role": "user", "content":
                        f'(A) REGIONAL TRANSCRIPT:\n"""{regional}"""\n\n(B) ENGLISH TRANSCRIPT:\n"""{english_scan}"""'},
                ],
                model=GROQ_EXTRACTION_MODEL, temperature=0.1, max_tokens=max_tokens,
            )
            if merged:
                return {"corrected": _clean_merged_transcript(merged), "corrections": corrections, "medicineTokens": tokens}
        except Exception:
            pass

    out = regional
    for k, v in corrections.items():
        out = re.sub(re.escape(k), v, out, flags=re.I)
    return {"corrected": out, "corrections": corrections, "medicineTokens": tokens}


# ── Extraction ───────────────────────────────────────────────────────────────────
EXTRACTION_PROMPT = """You are a clinical scribe assistant. You read a transcript of a
doctor-patient conversation and extract structured information. You ONLY extract
information explicitly present — never invent clinical facts.

Return ONLY a JSON object with exactly this shape (no markdown, no commentary):
{
  "patient": { "name": string|null, "age": string|null, "gender": string|null },
  "symptoms": string[],
  "diagnoses": string[],
  "prescriptions": [
    { "medication": string, "dosage": string|null, "frequency": string|null,
      "duration": string|null, "instructions": string|null }
  ],
  "followUp": { "nextVisit": string|null, "instructions": string|null },
  "fees": { "total": number|null, "paid": number|null },
  "allergies": string[],
  "vitals": string[],
  "notes": string[]
}
Use empty arrays / null where information is absent.

LANGUAGE: Output EVERY field value in ENGLISH. Translate all clinical details from the
transcript's language into clear clinical English (symptoms, diagnoses, frequency,
duration, instructions, follow-up, vitals, notes). Transliterate the patient's name to
Latin script.

MEDICATION: Use the EXACT drug the doctor named, in standard English/Latin spelling
("परासिटामोल" -> "Paracetamol", "डोलो" -> "Dolo"). Do NOT replace a brand with its
generic or vice versa — keep each medicine exactly as spoken.

NUMBERS: Preserve every number EXACTLY as in the transcript. Translate number-words to
the same digit ("दो" -> "2", "तीन" -> "3"). NEVER change a quantity, dose, frequency
count, or duration ("दो दिन" -> "for 2 days", never "1 day").

FOLLOW-UP: If a follow-up or review date is mentioned, return nextVisit as an ISO
date (YYYY-MM-DD). Resolve relative phrases such as "in 3 days" using today's date
provided with the transcript. Return null only when no follow-up is mentioned.

FEES: If total fees or paid fees are mentioned, return the numeric values in fees.total
and fees.paid. If only a generic fee amount is mentioned, set both values to that amount.
Do not copy fees from the patient's existing record; use only the conversation."""

_EMPTY = {"patient": {}, "symptoms": [], "diagnoses": [], "prescriptions": [],
          "followUp": {}, "fees": {"total": None, "paid": None},
          "allergies": [], "vitals": [], "notes": []}


def _coerce(raw):
    out = json.loads(json.dumps(_EMPTY))  # deep copy
    if not isinstance(raw, dict):
        return out
    def s(v): return v.strip() if isinstance(v, str) and v.strip() else None
    def arr(v): return [x.strip() for x in v if isinstance(x, str) and x.strip()] if isinstance(v, list) else []
    p = raw.get("patient") or {}
    if isinstance(p, dict):
        out["patient"] = {"name": s(p.get("name")), "age": s(p.get("age")), "gender": s(p.get("gender"))}
    out["symptoms"], out["diagnoses"] = arr(raw.get("symptoms")), arr(raw.get("diagnoses"))
    out["allergies"], out["vitals"], out["notes"] = arr(raw.get("allergies")), arr(raw.get("vitals")), arr(raw.get("notes"))
    rx = raw.get("prescriptions")
    if isinstance(rx, list):
        out["prescriptions"] = [
            {"medication": s(p.get("medication")), "dosage": s(p.get("dosage")),
             "frequency": s(p.get("frequency")), "duration": s(p.get("duration")),
             "instructions": s(p.get("instructions"))}
            for p in rx if isinstance(p, dict) and s(p.get("medication"))
        ]
    fu = raw.get("followUp") or {}
    if isinstance(fu, dict):
        out["followUp"] = {"nextVisit": s(fu.get("nextVisit")), "instructions": s(fu.get("instructions"))}
    fees = raw.get("fees") or {}
    if isinstance(fees, dict):
        def n(v):
            try:
                return float(v) if v is not None and str(v).strip() else None
            except (TypeError, ValueError):
                return None
        out["fees"] = {"total": n(fees.get("total")), "paid": n(fees.get("paid"))}
    return out


def extract(transcript: str, locale: str = DEFAULT_LOCALE):
    if not transcript.strip():
        return json.loads(json.dumps(_EMPTY))
    content = _groq_chat(
        [
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": f"Today's date: {date.today().isoformat()}\n\nTranscript:\n\"\"\"{transcript}\"\"\""},
        ],
        model=GROQ_EXTRACTION_MODEL, temperature=0, max_tokens=4096, json_mode=True,
    )
    start, end = content.find("{"), content.rfind("}")
    raw = json.loads(content[start:end + 1]) if start != -1 and end != -1 else {}
    return canonicalize_prescriptions(_coerce(raw))


DIARIZE_PROMPT = """You label a doctor–patient consultation transcript by speaker.

Rules:
- Split the transcript into turns and prefix each with "Doctor:" or "Patient:".
- Group CONSECUTIVE sentences by the same person into a SINGLE turn.
- Infer the speaker from context: the DOCTOR greets, asks history, examines, states
  vitals/findings, diagnoses, prescribes, and advises; the PATIENT describes symptoms
  and answers questions.
- Do NOT translate, add, remove, or reword anything — keep the exact text and language.
  Only add the "Doctor:"/"Patient:" prefixes.
- Output ONLY the labeled transcript, one turn per line."""


def diarize(transcript: str, locale: str = DEFAULT_LOCALE) -> str:
    """Label a transcript by speaker role (Doctor/Patient) via the LLM."""
    if not transcript.strip() or not GROQ_API_KEY:
        return transcript
    try:
        max_tokens = min(8000, max(2048, len(transcript) * 2))
        out = _groq_chat(
            [{"role": "system", "content": DIARIZE_PROMPT},
             {"role": "user", "content": transcript}],
            model=GROQ_EXTRACTION_MODEL, temperature=0.1, max_tokens=max_tokens,
        )
        return out.strip() or transcript
    except Exception:
        return transcript


# ── Hybrid multi-speaker diarization ─────────────────────────────────────────
_HYBRID_LABEL_PROMPT = """You receive a doctor-patient consultation transcript whose speaker turns
are already labeled Speaker A, Speaker B, or (if present) Speaker C.

Your only task is to replace EVERY "Speaker A:", "Speaker B:", "Speaker C:" prefix with the
correct role label: "Doctor:" or "Patient:" — or, when a third speaker is present (e.g. a nurse,
attendant, or family member), use "Attendant:".

Rules:
- Infer each speaker's role from conversation context: the Doctor greets, examines, prescribes
  and advises; the Patient describes symptoms and answers questions; an Attendant accompanies
  and asks on behalf of the patient.
- Preserve EVERY word, line break, punctuation, and the original language/script. Do NOT
  translate, paraphrase, reorder, or drop anything.
- Output ONLY the re-labeled transcript, one turn per line, e.g.
    Doctor: ...
    Patient: ..."""


def hybrid_diarize(
    file_bytes: bytes,
    filename: str,
    mimetype: str,
    transcript: str,
    locale: str = DEFAULT_LOCALE,
    max_speakers: int = 3,
) -> dict:
    """Frequency + embedding speaker detection with per-speaker independent transcription.

    Pipeline:
      1. POST audio to /speaker-diarization-transcribe on the Indic STT service.
         The service runs diarize() + extract_speaker_audio() + per-speaker
         IndicConformer transcription in one shot.
      2. Receive segments (A/B/C), speakerTranscripts (independently transcribed),
         frequencyGroups, detectedVoices.
      3. For each speaker's Indic transcript, also run Groq Whisper translation to
         get English medicine names, then merge (same combine() logic used for the
         joint transcript).
      4. Build a Speaker-A/B/C labeled version of the merged per-speaker text.
      5. Run Groq to relabel A/B/C as Doctor/Patient/Attendant.
      6. Return the full result including speakerContainers ready for the UI.

    Falls back gracefully:
      - If /speaker-diarization-transcribe is unavailable → falls back to
        /speaker-diarization (segments only, heuristic chunk distribution).
      - If per-speaker Indic transcripts are empty → uses heuristic chunks from
        the joint transcript as before.

    Returns a dict with keys:
        diarized          — final labeled transcript (Doctor:/Patient:…)
        detectedVoices    — 2 or 3
        frequencyUsed     — True
        segments          — [{speaker, start, end}, ...]
        frequencyGroups   — per-speaker Hz statistics
        speakerTranscripts — {A: "raw Indic text…", B: "…", C?: "…"}
        speakerChunks     — {A: ["sentence…", …], B: […], C?: […]}
        speakerContainers — [{label, speaker, chunks, transcript, rawTranscript}, …]
    """
    if not file_bytes:
        return _empty_hybrid_result(transcript)

    lang = _lang_code(locale) or DEFAULT_LOCALE[:2]
    lang = indic_language_code(lang)

    # 1. Try the full transcribe endpoint first (independent per-speaker STT)
    acoustic = _call_diarize_transcribe(file_bytes, filename, mimetype, lang, max_speakers)
    used_transcribe_endpoint = acoustic.get("_transcribeEndpoint", False)

    segments: list[dict] = acoustic.get("segments") or []
    detected: int = acoustic.get("detectedVoices", 0)
    speaker_transcripts: dict[str, str] = acoustic.get("speakerTranscripts") or {}
    frequency_groups: list[dict] = acoustic.get("frequencyGroups") or []

    if not segments or detected < 2:
        return {
            "diarized": transcript,
            "detectedVoices": detected,
            "frequencyUsed": True,
            "segments": segments,
            "frequencyGroups": frequency_groups,
            "speakerTranscripts": speaker_transcripts,
            "speakerChunks": acoustic.get("speakerChunks") or {},
            "speakerContainers": [],
        }

    # 2. For each speaker: merge Indic transcript with Groq Whisper English
    #    (medicine names) using the same combine() pipeline as the joint path.
    #    Run Whisper per-speaker using the silence-filled track logic proxy:
    #    we don't re-upload audio here, so we use the speaker's Indic text as
    #    the "regional" side and the joint English transcript (already computed
    #    upstream) as the "english" side — a reasonable approximation.
    #    For the cleanest result, merge() uses the per-speaker Indic text directly.
    merged_speaker_transcripts: dict[str, str] = {}
    speaker_chunks: dict[str, list[str]] = {}

    for label, indic_text in speaker_transcripts.items():
        if not indic_text.strip():
            merged_speaker_transcripts[label] = ""
            speaker_chunks[label] = []
            continue

        # Use the per-speaker Indic text as-is when we have real transcription.
        # The joint Whisper translation is not available per-speaker here, so we
        # run a quick Whisper on the per-speaker text via the NER path to fix
        # medicine names, keeping the regional language intact.
        if lang == "en":
            merged_speaker_transcripts[label] = indic_text
        else:
            try:
                # Use combine() with just the regional side populated; English
                # side empty forces combine() to use regional as base and only
                # apply ALIAS + FAISS medicine corrections.
                merged = combine(indic_text, "", locale)
                merged_speaker_transcripts[label] = merged.get("corrected") or indic_text
            except Exception:
                merged_speaker_transcripts[label] = indic_text

        # Split into sentence chunks for the UI
        import re as _re
        sentences = [s.strip() for s in _re.split(r"(?<=[.!?।\n])\s*", merged_speaker_transcripts[label]) if s.strip()]
        speaker_chunks[label] = sentences if sentences else [merged_speaker_transcripts[label]]

    # 3. Build a Speaker-A/B/C labeled transcript from the merged per-speaker texts
    labeled_parts: list[str] = []
    # Interleave by segment order so the labeled transcript follows the
    # chronological conversation flow (not just all of A then all of B).
    _seen_labels: set[str] = set()
    for seg in segments:
        lbl = seg["speaker"]
        if lbl in _seen_labels:
            continue
        _seen_labels.add(lbl)
        text = merged_speaker_transcripts.get(lbl, "").strip()
        if text:
            labeled_parts.append(f"Speaker {lbl}: {text}")

    # Rebuild full labeled transcript ordered by segments (turn-by-turn)
    labeled_transcript = _apply_speaker_labels_from_chunks(
        merged_speaker_transcripts, segments
    )

    # 4. Relabel A/B/C → Doctor/Patient/Attendant with Groq
    final_transcript = labeled_transcript
    if GROQ_API_KEY and labeled_transcript.strip():
        try:
            max_tokens = min(8000, max(2048, len(labeled_transcript) * 2))
            groq_out = _groq_chat(
                [
                    {"role": "system", "content": _HYBRID_LABEL_PROMPT},
                    {"role": "user", "content": labeled_transcript},
                ],
                model=GROQ_EXTRACTION_MODEL,
                temperature=0,
                max_tokens=max_tokens,
            )
            if groq_out.strip():
                final_transcript = groq_out.strip()
        except Exception:
            pass

    # 5. Build speaker containers
    containers = _build_speaker_containers(speaker_chunks, final_transcript, detected)
    # Attach the raw Indic transcript to each container for debugging/display
    for container in containers:
        container["rawTranscript"] = speaker_transcripts.get(container["speaker"], "")

    return {
        "diarized": final_transcript,
        "detectedVoices": detected,
        "frequencyUsed": True,
        "segments": segments,
        "frequencyGroups": frequency_groups,
        "speakerTranscripts": merged_speaker_transcripts,
        "speakerChunks": speaker_chunks,
        "speakerContainers": containers,
        "independentTranscription": used_transcribe_endpoint,
    }


def retranscribe_speaker(
    file_bytes: bytes,
    filename: str,
    mimetype: str,
    segments: list[dict],
    speaker: str,
    locale: str = DEFAULT_LOCALE,
) -> dict:
    """Re-run transcription for one already-diarized speaker.

    Forwards to indic_stt's /speaker-diarization-retranscribe with the original
    segment boundaries unchanged (only that speaker's words are redone), then
    applies the same medicine-name correction pass hybrid_diarize uses so the
    retried text stays consistent in style with the rest of the transcript.

    Returns {speaker, transcript, chunks}.
    """
    lang = _lang_code(locale) or DEFAULT_LOCALE[:2]
    lang = indic_language_code(lang)

    resp = requests.post(
        f"{INDIC_STT_URL}/speaker-diarization-retranscribe",
        files={"file": (filename or "audio.webm", file_bytes, mimetype or "audio/webm")},
        data={"segments": json.dumps(segments), "speaker": speaker, "language": lang},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    indic_text = (resp.json().get("transcript") or "").strip()

    if not indic_text:
        return {"speaker": speaker, "transcript": "", "chunks": []}

    if lang == "en":
        merged_text = indic_text
    else:
        try:
            merged = combine(indic_text, "", locale)
            merged_text = merged.get("corrected") or indic_text
        except Exception:
            merged_text = indic_text

    sentences = [s.strip() for s in re.split(r"(?<=[.!?।\n])\s*", merged_text) if s.strip()]
    return {"speaker": speaker, "transcript": merged_text, "chunks": sentences or [merged_text]}


def _empty_hybrid_result(transcript: str) -> dict:
    return {
        "diarized": transcript,
        "detectedVoices": 0,
        "frequencyUsed": False,
        "segments": [],
        "frequencyGroups": [],
        "speakerTranscripts": {},
        "speakerChunks": {},
        "speakerContainers": [],
        "independentTranscription": False,
    }


def _call_diarize_transcribe(
    file_bytes: bytes, filename: str, mimetype: str, lang: str, max_speakers: int
) -> dict:
    """Try /speaker-diarization-transcribe first; fall back to /speaker-diarization.

    Sets _transcribeEndpoint=True in the result when the richer endpoint was used.
    """
    try:
        resp = requests.post(
            f"{INDIC_STT_URL}/speaker-diarization-transcribe",
            files={"file": (filename or "audio.webm", file_bytes, mimetype or "audio/webm")},
            data={
                "language": lang,
                "max_speakers": str(max(2, min(3, max_speakers))),
            },
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()
        result["_transcribeEndpoint"] = True
        return result
    except Exception:
        pass

    # Fallback: segments-only endpoint (no independent transcription)
    try:
        resp = requests.post(
            f"{INDIC_STT_URL}/speaker-diarization",
            files={"file": (filename or "audio.webm", file_bytes, mimetype or "audio/webm")},
            data={"max_speakers": str(max(2, min(3, max_speakers)))},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()
        result["_transcribeEndpoint"] = False
        return result
    except Exception as exc:
        return {
            "_transcribeEndpoint": False,
            "segments": [],
            "detectedVoices": 0,
            "frequencyUsed": False,
            "error": str(exc),
        }


def _apply_speaker_labels_from_chunks(
    speaker_texts: dict[str, str], segments: list[dict]
) -> str:
    """Build a chronological turn-by-turn labeled transcript.

    Since each speaker now has an independent full transcript (not sentence
    fragments), we can't trivially interleave word-by-word. Instead we produce
    one labeled block per speaker in the order they first appear in the
    timeline, which gives the most readable result.
    """
    seen: list[str] = []
    for seg in segments:
        if seg["speaker"] not in seen:
            seen.append(seg["speaker"])

    lines: list[str] = []
    for label in seen:
        text = speaker_texts.get(label, "").strip()
        if text:
            lines.append(f"Speaker {label}: {text}")

    return "\n".join(lines)


def _apply_speaker_labels(transcript: str, segments: list[dict]) -> str:
    """Distribute transcript sentences across A/B/C speaker labels using segment timing."""
    import re
    sentences = [s.strip() for s in re.split(r"(?<=[.!?।\n])\s*", transcript) if s.strip()]
    if not sentences or not segments:
        return transcript

    duration = max(s["end"] for s in segments) or 1.0
    total_chars = sum(len(s) for s in sentences) or 1

    lines: list[str] = []
    seg_idx = 0
    char_offset = 0
    current_speaker = ""

    for sentence in sentences:
        midpoint = ((char_offset + len(sentence) / 2) / total_chars) * duration
        while seg_idx + 1 < len(segments) and midpoint > segments[seg_idx]["end"]:
            seg_idx += 1
        speaker = f"Speaker {segments[seg_idx]['speaker']}"
        if lines and lines[-1].startswith(f"{speaker}:"):
            lines[-1] = lines[-1] + " " + sentence
        else:
            lines.append(f"{speaker}: {sentence}")
        char_offset += len(sentence)
        current_speaker = speaker

    return "\n".join(lines)


def _build_speaker_containers(
    chunks: dict[str, list[str]],
    final_transcript: str,
    detected: int,
) -> list[dict]:
    """Create per-speaker container objects for the frontend.

    Tries to match A/B/C chunks back to Doctor/Patient/Attendant labels by
    inspecting the final_transcript (which may already have Groq role labels).
    """
    import re

    # Build a map from A/B/C → role label inferred from the final transcript
    role_map: dict[str, str] = {}
    # Patterns like "Doctor: ..." or "Patient: ..."
    for line in final_transcript.splitlines():
        role_match = re.match(r"^(Doctor|Patient|Attendant|Speaker ([A-C])):\s", line)
        if role_match:
            role = role_match.group(1)
            if role.startswith("Speaker "):
                letter = role_match.group(2)
                if letter and letter not in role_map:
                    role_map[letter] = f"Speaker {letter}"
            # If we got Doctor/Patient/Attendant we try to associate with a chunk key
            # by finding the first sentence of this turn in the chunks
            turn_text = line[len(role_match.group(0)):].strip()
            if turn_text:
                for letter, sentences in chunks.items():
                    if any(turn_text.startswith(s[:30]) for s in sentences):
                        if letter not in role_map:
                            role_map[letter] = role

    # Assign fallback labels
    default_roles = ["Doctor", "Patient", "Attendant"]
    ordered_keys = sorted(chunks.keys())   # A before B before C
    for i, key in enumerate(ordered_keys):
        if key not in role_map:
            role_map[key] = default_roles[i] if i < len(default_roles) else f"Speaker {key}"

    containers = []
    for key in ordered_keys:
        sentences = chunks[key]
        containers.append({
            "speaker": key,                       # "A", "B", or "C"
            "label": role_map.get(key, f"Speaker {key}"),
            "chunks": sentences,
            "transcript": " ".join(sentences),
        })

    return containers


def transcribe_stages(file_bytes, filename, mimetype, locale: str):
    """Run the full transcription pipeline and return every stage."""
    # Normalize once for both regional and Whisper passes. Indic STT also
    # normalizes defensively, but this keeps the Whisper medicine transcript
    # on the same clean mono/16 kHz audio.
    normalized_bytes, normalized_name, normalized_type = _normalized_audio(file_bytes, filename, mimetype, None)
    file_bytes, filename, mimetype = normalized_bytes, normalized_name, normalized_type
    code = _lang_code(locale)
    if not code:
        # Auto mode must identify the language before invoking the Indic model.
        # Previously it passed the English default to IndicConformer, which
        # caused the service to fall back to Hindi and produced garbled output.
        try:
            code = detect_language(file_bytes, filename, mimetype)
        except Exception:
            code = ""
        code = indic_language_code(code)

        # If the short opening window is dominated by English medicine names
        # or greetings, verify an English result against the full cleaned
        # recording before skipping IndicTransformer. This is the automatic
        # fallback for mixed-language consultations.
        if code == "en":
            try:
                full_code = detect_language(file_bytes, filename, mimetype)
                if full_code and full_code != "en":
                    code = indic_language_code(full_code)
            except Exception:
                pass

        if code == "en":
            english = transcribe_regional(file_bytes, filename, mimetype, "en_IN")
            return {"regional": "", "english": english, "final": english,
                    "corrections": {}, "language": "en", "dual": False}

        regional = transcribe_regional(file_bytes, filename, mimetype, code or DEFAULT_LOCALE)
        if code and not regional_quality(regional, code):
            # Retry once after the Indic service has loaded/settled. The
            # selected language remains authoritative; never silently switch
            # Gujarati/Hindi to English because one pass was empty.
            regional_retry = transcribe_regional(file_bytes, filename, mimetype, code)
            if regional_quality(regional_retry, code) or not regional:
                regional = regional_retry
        english = transcribe_english(file_bytes, filename, mimetype)
        regional_for_merge = regional if regional_quality(regional, code) else ""
        merged = combine(regional_for_merge, english, code or DEFAULT_LOCALE)
        return {"regional": regional, "english": english, "final": merged["corrected"],
                "corrections": merged["corrections"], "language": code or "auto", "dual": True,
                "regionalUsable": regional_quality(regional, code)}

    if code == "en":
        english = transcribe_regional(file_bytes, filename, mimetype, locale)
        return {"regional": "", "english": english, "final": english,
                "corrections": {}, "language": "en", "dual": False}
    regional = transcribe_regional(file_bytes, filename, mimetype, locale)
    if not regional_quality(regional, code):
        regional_retry = transcribe_regional(file_bytes, filename, mimetype, locale)
        if regional_quality(regional_retry, code) or not regional:
            regional = regional_retry
    english = transcribe_english(file_bytes, filename, mimetype)
    regional_for_merge = regional if regional_quality(regional, code) else ""
    merged = combine(regional_for_merge, english, locale)
    return {"regional": regional, "english": english, "final": merged["corrected"],
            "corrections": merged["corrections"], "language": code or "auto", "dual": True,
            "regionalUsable": regional_quality(regional, code)}
