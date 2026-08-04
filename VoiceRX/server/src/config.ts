import { config as loadDotenv } from "dotenv";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// Load the repo-root .env no matter where the server is started from.
// This file lives at <root>/server/src/config.ts (dev) or <root>/server/dist/config.js
// (prod) — both are two levels below the repo root.
const here = dirname(fileURLToPath(import.meta.url));
const candidates = [resolve(here, "..", "..", ".env"), resolve(process.cwd(), ".env")];
for (const path of candidates) {
  if (existsSync(path)) {
    loadDotenv({ path });
    break;
  }
}

export type ExtractionProvider = "heuristic" | "openai" | "bedrock" | "groq";

export const config = {
  host: process.env.HOST ?? "127.0.0.1",
  port: Number(process.env.PORT ?? 4000),
  clientOrigins:
    process.env.CLIENT_ORIGIN === "*"
      ? true
      : (process.env.CLIENT_ORIGIN ?? "http://localhost:5173")
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean),

  extractionProvider: (process.env.EXTRACTION_PROVIDER ?? "heuristic") as ExtractionProvider,

  openai: {
    apiKey: process.env.OPENAI_API_KEY ?? "",
    baseUrl: process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1",
    model: process.env.OPENAI_MODEL ?? "gpt-4o-mini",
  },

  bedrock: {
    region: process.env.AWS_REGION ?? "us-east-1",
    modelId: process.env.BEDROCK_MODEL_ID ?? "anthropic.claude-3-5-sonnet-20240620-v1:0",
  },

  // ── STT transcript correction (ported from groq_transcript_processor.dart) ──
  correction: {
    // Master switch. When on, /api/extract cleans the transcript first.
    enabled: (process.env.CORRECTION_ENABLED ?? "true").toLowerCase() !== "false",
    // STT locale passed to the Indian-OPD correction prompt (en_IN, hi_IN, gu_IN…).
    locale: process.env.CORRECTION_LOCALE ?? "en_IN",
  },

  // Groq (OpenAI-compatible) used for medicine NER + transcript cleanup,
  // and (by default) for Whisper audio transcription.
  groq: {
    apiKey: process.env.GROQ_API_KEY ?? "",
    baseUrl: process.env.GROQ_BASE_URL ?? "https://api.groq.com/openai/v1",
    // Fast model for the correction NER/cleanup passes.
    model: process.env.GROQ_MODEL ?? "llama-3.1-8b-instant",
    // Stronger model for structured medical extraction (EXTRACTION_PROVIDER=groq).
    extractionModel: process.env.GROQ_EXTRACTION_MODEL ?? "llama-3.3-70b-versatile",
    // Whisper model for audio transcription (/api/transcribe). Tried
    // "whisper-large-v3-turbo" for speed; reverted — on real Indian-accented,
    // code-switched audio it hallucinated repeated words and mid-transcript
    // switched from Devanagari to English. Not worth the latency win here.
    transcriptionModel: process.env.GROQ_TRANSCRIPTION_MODEL ?? "whisper-large-v3",
  },

  // ── Audio transcription provider (pluggable) ──────────────────────────────
  //   groq     = Groq Whisper (default; uses the groq.* settings above)
  //   bhashini = Govt of India Bhashini/ULCA (free; runs AI4Bharat IndicConformer)
  //   custom   = any OpenAI-compatible /audio/transcriptions server, e.g. a
  //              self-hosted IndicWhisper via faster-whisper-server / speaches.
  transcription: {
    provider: (process.env.TRANSCRIPTION_PROVIDER ?? "groq") as "groq" | "custom" | "bhashini",
    baseUrl: process.env.TRANSCRIPTION_BASE_URL ?? "",
    apiKey: process.env.TRANSCRIPTION_API_KEY ?? "",
    model: process.env.TRANSCRIPTION_MODEL ?? "",
    // Dual-ASR: also run Groq Whisper (→English) in parallel with the regional
    // provider and merge, so medicine names are resolved in English via FAISS.
    // Requires a non-groq regional provider + GROQ_API_KEY.
    dual: (process.env.DUAL_ASR ?? "false").toLowerCase() === "true",
  },

  // Bhashini / ULCA (https://bhashini.gov.in) — free Indic ASR (IndicConformer).
  // Get userID + ulcaApiKey from the "My Profile" section after registering.
  bhashini: {
    userId: process.env.BHASHINI_USER_ID ?? "",
    ulcaApiKey: process.env.BHASHINI_ULCA_API_KEY ?? "",
    // Public MeitY pipeline id (override if you use a different one).
    pipelineId: process.env.BHASHINI_PIPELINE_ID ?? "64392f96daac500b55c543cd",
    configUrl:
      process.env.BHASHINI_CONFIG_URL ??
      "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline",
  },

  // FastAPI FAISS medicine-search service (medicine_pipeline/server.py).
  medicineSearch: {
    url: process.env.MEDICINE_SEARCH_URL ?? "http://localhost:8000",
    // Min RapidFuzz/FAISS score to treat a match as high-confidence.
    confidenceThreshold: Number(process.env.MEDICINE_CONFIDENCE_THRESHOLD ?? 85),
    timeoutMs: Number(process.env.MEDICINE_SEARCH_TIMEOUT_MS ?? 4000),
  },
};
