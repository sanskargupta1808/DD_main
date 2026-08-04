import { config } from "../config.js";
import { bhashiniTranscribe } from "./bhashini.js";
import { combineRegionalAndEnglish } from "./correction.js";

export interface TranscribeResult {
  transcript: string;
  model: string;
  provider: string;
  /** Whisper/STT language code actually used (omitted = auto). */
  language?: string;
  /** True when the result is already STT-corrected (dual merge) — skip re-correction. */
  corrected?: boolean;
  /** Medicine-name corrections applied during the dual merge. */
  corrections?: Record<string, string>;
}

const MEDICAL_PROMPT =
  "Indian outpatient clinic dictation. Common terms: Paracetamol, Dolo 650, " +
  "Calpol, Combiflam, Kalcoral D, Augmentin, Azithromycin, Amoxicillin, " +
  "Pantoprazole, Pan-D, Rabeprazole, Zerodol, Crocin, Montek LC, Allegra, " +
  "Metformin, Telma, Amlodipine, Atorvastatin, Ecosprin, Ondansetron, " +
  "Domstal, mg, ml, tablet, capsule, syrup, BD, OD, TDS, once daily, twice daily.";

const SUPPORTED_LANGS = new Set([
  "en", "hi", "gu", "mr", "bn", "ta", "te", "kn", "ml", "pa", "or", "as", "ur",
]);

export function resolveLang(localeOrLang?: string): string | undefined {
  if (!localeOrLang || localeOrLang.toLowerCase() === "auto") return undefined;
  const code = localeOrLang.slice(0, 2).toLowerCase();
  return SUPPORTED_LANGS.has(code) ? code : undefined;
}

interface Target {
  baseUrl: string;
  apiKey: string;
  model: string;
  provider: string;
}

function groqTarget(): Target {
  return {
    baseUrl: config.groq.baseUrl.replace(/\/$/, ""),
    apiKey: config.groq.apiKey,
    model: config.groq.transcriptionModel,
    provider: "groq",
  };
}

function customTarget(): Target {
  return {
    baseUrl: config.transcription.baseUrl.replace(/\/$/, ""),
    apiKey: config.transcription.apiKey,
    model: config.transcription.model || "whisper-1",
    provider: "custom",
  };
}

/**
 * Call an OpenAI-compatible audio endpoint.
 * translate=true → /audio/translations (any language → English).
 * translate=false → /audio/transcriptions (in `lang`).
 */
async function openAICompatibleCall(
  buffer: Buffer,
  filename: string,
  mimetype: string,
  lang: string | undefined,
  target: Target,
  translate: boolean
): Promise<TranscribeResult> {
  if (!target.baseUrl) {
    throw new Error("Transcription baseUrl is not configured.");
  }
  if (target.provider === "groq" && !target.apiKey) {
    throw new Error("GROQ_API_KEY is not set — cannot transcribe audio.");
  }

  const form = new FormData();
  form.append("file", new Blob([buffer], { type: mimetype || "audio/webm" }), filename || "audio.webm");
  form.append("model", target.model);
  form.append("prompt", MEDICAL_PROMPT);
  form.append("temperature", "0");
  form.append("response_format", "json");
  if (!translate && lang) form.append("language", lang);

  const headers: Record<string, string> = {};
  if (target.apiKey) headers.Authorization = `Bearer ${target.apiKey}`;

  const endpoint = translate ? "/audio/translations" : "/audio/transcriptions";
  const res = await fetch(`${target.baseUrl}${endpoint}`, { method: "POST", headers, body: form });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Transcription (${target.provider}${translate ? "/translate" : ""}) error ${res.status}: ${body.slice(0, 200)}`);
  }
  const data: any = await res.json();
  return {
    transcript: (data?.text ?? "").trim(),
    model: target.model,
    provider: target.provider,
    language: translate ? "en" : lang,
  };
}

/** Single-provider transcription in the configured/selected language. */
async function regionalTranscribe(
  buffer: Buffer,
  filename: string,
  mimetype: string,
  localeOrLang?: string
): Promise<TranscribeResult> {
  const lang = resolveLang(localeOrLang);
  // The Indic providers (IndicConformer/Bhashini) cannot transcribe English —
  // use Groq Whisper for English so the transcript is actually English.
  if (lang === "en" && config.groq.apiKey && config.transcription.provider !== "groq") {
    return openAICompatibleCall(buffer, filename, mimetype, "en", groqTarget(), false);
  }
  if (config.transcription.provider === "bhashini") {
    return bhashiniTranscribe(buffer, mimetype, localeOrLang ?? config.correction.locale);
  }
  const target = config.transcription.provider === "custom" ? customTarget() : groqTarget();
  return openAICompatibleCall(buffer, filename, mimetype, lang, target, false);
}

/** Groq Whisper translate → English (used as the medicine-name source in dual mode). */
async function groqEnglish(buffer: Buffer, filename: string, mimetype: string): Promise<TranscribeResult> {
  return openAICompatibleCall(buffer, filename, mimetype, undefined, groqTarget(), true);
}

/**
 * Groq Whisper TRANSCRIBE (not translate) with English forced as the decoding
 * language. Genuine English speech comes through as real English; genuine
 * Hindi/regional speech comes through as rough phonetic Latin-script
 * transliteration, NOT a translation — same word order, same content, just
 * mis-rendered script. That word-order preservation is exactly what makes
 * this useful as a reference for acousticDiarize's code-switch correction:
 * an LLM reading this alongside the primary (possibly all-Devanagari)
 * transcript can tell which Devanagari words were actually spoken in
 * English, because they show up as real English here at the same point in
 * the conversation — translate mode would rephrase everything, losing that
 * word-for-word correspondence.
 */
export async function transcribeEnglishForced(
  buffer: Buffer,
  filename: string,
  mimetype: string,
): Promise<string> {
  const result = await openAICompatibleCall(buffer, filename, mimetype, "en", groqTarget(), false);
  return result.transcript;
}

export interface WordTimestamp {
  word: string;
  start: number;
  end: number;
}

/**
 * Whole-recording Groq Whisper transcription with word-level timestamps
 * (used by acousticDiarize to attribute words to speakers by real timing
 * instead of transcribing each speaker turn in isolation — short isolated
 * clips reliably make Whisper hallucinate fluent nonsense; a single pass
 * over the full recording has the context to transcribe reliably).
 *
 * Uses /audio/transcriptions (not /translations), so multilingual/code-switched
 * speech comes back in its original language/script — this is a genuine
 * transcription, not an English translation.
 */
export async function transcribeWithWordTimestamps(
  buffer: Buffer,
  filename: string,
  mimetype: string,
  localeOrLang?: string,
): Promise<{ text: string; words: WordTimestamp[]; language?: string }> {
  if (!config.groq.apiKey) {
    throw new Error("GROQ_API_KEY is not set — word-timestamp alignment requires Groq Whisper.");
  }
  const lang = resolveLang(localeOrLang);
  const form = new FormData();
  form.append("file", new Blob([buffer], { type: mimetype || "audio/webm" }), filename || "audio.webm");
  form.append("model", config.groq.transcriptionModel);
  // No `prompt` here (unlike openAICompatibleCall) — a text prompt without an
  // explicit `language` reliably leaked verbatim into the transcript on this
  // whole-file, often-long, sometimes code-switched pass (confirmed: it
  // returned "Common terms. Common terms." then a raw dump of the drug list
  // from MEDICAL_PROMPT). Medicine-name accuracy is handled afterward by
  // correctLabeledTranscript's NER+FAISS pass instead.
  form.append("temperature", "0");
  form.append("response_format", "verbose_json");
  form.append("timestamp_granularities[]", "word");
  if (lang) form.append("language", lang);

  const baseUrl = config.groq.baseUrl.replace(/\/$/, "");
  const res = await fetch(`${baseUrl}/audio/transcriptions`, {
    method: "POST",
    headers: { Authorization: `Bearer ${config.groq.apiKey}` },
    body: form,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Groq word-timestamp transcription error ${res.status}: ${body.slice(0, 200)}`);
  }
  const data: any = await res.json();
  const words: WordTimestamp[] = (data?.words ?? []).map((w: any) => ({
    word: String(w.word ?? "").trim(),
    start: Number(w.start ?? 0),
    end: Number(w.end ?? 0),
  })).filter((w: WordTimestamp) => w.word);
  return { text: (data?.text ?? "").trim(), words, language: data?.language };
}

/**
 * Dual-ASR: regional provider (accurate conversation) + Groq Whisper→English
 * (medicine names) in parallel, then Groq merges them — medicine names resolved
 * in English via FAISS, conversation kept in the regional language.
 */
async function transcribeDual(
  buffer: Buffer,
  filename: string,
  mimetype: string,
  localeOrLang?: string
): Promise<TranscribeResult> {
  // English selected: there's no regional language to preserve — the English
  // (Whisper) transcript IS the final. Skip IndicConformer + merge.
  if (resolveLang(localeOrLang) === "en") {
    const en = await openAICompatibleCall(buffer, filename, mimetype, "en", groqTarget(), false);
    return { ...en, provider: "groq(english)", corrected: false, corrections: {} };
  }

  const [regional, english] = await Promise.all([
    regionalTranscribe(buffer, filename, mimetype, localeOrLang),
    groqEnglish(buffer, filename, mimetype),
  ]);

  const merged = await combineRegionalAndEnglish(
    regional.transcript,
    english.transcript,
    localeOrLang ?? config.correction.locale
  );

  return {
    transcript: merged.corrected,
    model: `${regional.model}+${config.groq.transcriptionModel}`,
    provider: `dual(${regional.provider}+groq)`,
    language: regional.language,
    corrected: merged.applied,
    corrections: merged.corrections,
  };
}

export async function transcribeAudio(
  buffer: Buffer,
  filename: string,
  mimetype: string,
  localeOrLang?: string
): Promise<TranscribeResult> {
  // Dual mode needs a non-groq regional provider + a Groq key for the English pass.
  if (config.transcription.dual && config.transcription.provider !== "groq" && config.groq.apiKey) {
    return transcribeDual(buffer, filename, mimetype, localeOrLang);
  }
  return regionalTranscribe(buffer, filename, mimetype, localeOrLang);
}

export type TranscribeMode = "auto" | "regional" | "english";

/**
 * Mode-aware entry point so the client can fetch the dual-ASR stages separately
 * and display each as it arrives:
 *   - "regional" → only the regional provider (IndicConformer/Bhashini/custom)
 *   - "english"  → only Groq Whisper (→English)
 *   - "auto"     → the configured behaviour (dual merge or single)
 */
export async function transcribeWithMode(
  buffer: Buffer,
  filename: string,
  mimetype: string,
  localeOrLang: string | undefined,
  mode: TranscribeMode
): Promise<TranscribeResult> {
  if (mode === "regional") return regionalTranscribe(buffer, filename, mimetype, localeOrLang);
  if (mode === "english") return groqEnglish(buffer, filename, mimetype);
  return transcribeAudio(buffer, filename, mimetype, localeOrLang);
}
