import type { ExtractResponse } from "./types";

const API_BASE = "/api";
export const FLASK_API_BASE = import.meta.env.VITE_FLASK_API_URL ?? "http://127.0.0.1:5005";

export interface TranscribeResult {
  transcript: string;
  model: string;
  provider?: string;
  /** True when the transcript is already STT-corrected (dual merge). */
  corrected?: boolean;
  corrections?: Record<string, string>;
}

export async function transcribeAudio(
  blob: Blob,
  locale?: string,
  mode?: "regional" | "english" | "auto"
): Promise<TranscribeResult> {
  const form = new FormData();
  const ext = blob.type.includes("mp4") ? "m4a" : blob.type.includes("ogg") ? "ogg" : "webm";
  form.append("audio", blob, `recording.${ext}`);
  if (locale) form.append("locale", locale);
  if (mode) form.append("mode", mode);
  const res = await fetch(`${API_BASE}/transcribe`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`Transcription failed (${res.status}). ${detail}`);
  }
  return res.json();
}

export interface CombineResult {
  corrected: string;
  corrections: Record<string, string>;
  medicineTokens: string[];
  applied: boolean;
  note?: string;
}

export async function combineTranscripts(
  regional: string,
  english: string,
  locale?: string
): Promise<CombineResult> {
  const res = await fetch(`${API_BASE}/combine`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ regional, english, locale }),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`Combine failed (${res.status}). ${detail}`);
  }
  return res.json();
}

export interface HealthInfo {
  transcription?: { provider?: string; dual?: boolean };
}

export interface VoiceSegment {
  speaker: "A" | "B" | "C";
  start: number;
  end: number;
  text?: string;
}

export interface FrequencyGroup {
  speaker: "A" | "B" | "C";
  minHz: number;
  maxHz: number;
  meanHz: number;
  samples: number;
}

/** One person's slice of the consultation — returned by hybrid diarization. */
export interface SpeakerContainer {
  /** Raw cluster label A / B / C */
  speaker: "A" | "B" | "C";
  /** Human-readable role: Doctor, Patient, or Attendant */
  label: string;
  /** Individual sentences attributed to this speaker */
  chunks: string[];
  /** Full concatenated transcript for this speaker (medicine-corrected) */
  transcript: string;
  /** Raw Indic STT output before medicine correction (debug / display) */
  rawTranscript?: string;
}

export interface DiarizeResult {
  diarized: string;
  detectedVoices?: number;
  frequencyUsed?: boolean;
  segments?: VoiceSegment[];
  frequencyGroups?: FrequencyGroup[];
  /** Raw per-speaker sentence buckets keyed by A/B/C */
  speakerChunks?: Record<string, string[]>;
  /** Independently transcribed text per speaker (high-quality path) */
  speakerTranscripts?: Record<string, string>;
  /** Ready-to-render per-speaker containers (hybrid and acoustic modes) */
  speakerContainers?: SpeakerContainer[];
  /** True when each speaker was transcribed independently (vs heuristic split) */
  independentTranscription?: boolean;
  /** acoustic mode only: medicine-name corrections applied post-labeling */
  corrections?: Record<string, string>;
  /** acoustic mode only: medicine tokens detected by Groq NER */
  medicineTokens?: string[];
}

export async function diarizeTranscript(
  transcript: string,
  locale?: string,
  mode: "ai" | "hybrid" | "acoustic" = "ai",
  audio?: Blob,
  maxSpeakers: 2 | 3 = 3,
): Promise<DiarizeResult> {
  const form = new FormData();
  if (audio) form.append("audio", audio, "recording.webm");
  // Acoustic mode transcribes each detected turn independently — it never
  // reads the joint transcript text, but the field is still required by the
  // route's validation for the other modes, so send it regardless.
  form.append("transcript", transcript);
  if (locale) form.append("locale", locale);
  form.append("mode", mode);
  if (mode === "hybrid" || mode === "acoustic") form.append("max_speakers", String(maxSpeakers));
  const res = await fetch(`${API_BASE}/diarize`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`Diarization failed (${res.status}). ${detail}`);
  }
  const data = (await res.json()) as DiarizeResult;
  return { ...data, diarized: data.diarized ?? transcript };
}

export async function fetchHealth(): Promise<HealthInfo> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed (${res.status}).`);
  return res.json();
}

export interface ImportedSession {
  session_id: string;
  audio_id: string;
  audio_url: string;
  locale: string;
  transcript: string;
  regional: string;
  english: string;
  final: string;
  language?: string;
  regionalUsable?: boolean;
  provider?: string;
  extraction?: ExtractResponse["extraction"] | null;
  corrections?: Record<string, string>;
}

export async function fetchImportedSession(sessionId: string): Promise<ImportedSession> {
  const res = await fetch(`${FLASK_API_BASE}/api/session/${encodeURIComponent(sessionId)}`);
  if (!res.ok) throw new Error(`VoiceRX session load failed (${res.status}).`);
  return res.json();
}

export async function fetchImportedAudio(audioUrl: string): Promise<Blob> {
  const res = await fetch(`${FLASK_API_BASE}${audioUrl}`);
  if (!res.ok) throw new Error(`VoiceRX audio load failed (${res.status}).`);
  return res.blob();
}

export async function extractMedical(
  transcript: string,
  locale?: string,
  correct?: boolean
): Promise<ExtractResponse> {
  const body: Record<string, unknown> = { transcript };
  if (locale) body.locale = locale;
  if (correct === false) body.correct = false;
  const res = await fetch(`${API_BASE}/extract`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`Extraction request failed (${res.status}). ${detail}`);
  }
  return res.json();
}
