import { Agent } from "undici";
import { config } from "../config.js";
import type { Segment } from "./frequencyDiarize.js";

// undici's default headers/body timeout is 300s. A long recording — or an
// indic_stt process that's been holding pyannote + IndicConformer + resemblyzer
// in memory for a while and has slowed down — can legitimately take longer
// than that; observed it happen more than once in practice. This is a local
// service call with no user waiting on a browser timeout shorter than this,
// so there's no downside to being generous here.
const longRunningAgent = new Agent({ headersTimeout: 900_000, bodyTimeout: 900_000 });

/**
 * Real acoustic speaker-turn boundaries from indic_stt's pyannote engine —
 * no transcript involved, no LLM guessing. Boundaries only; callers
 * transcribe each turn independently (see acousticDiarize in diarize.ts).
 */
export async function pyannoteDiarize(
  audio: Buffer,
  filename: string,
  mimetype: string,
  maxSpeakers: 2 | 3 = 3,
): Promise<{ segments: Segment[]; detectedVoices: number }> {
  const form = new FormData();
  form.append("file", new Blob([audio], { type: mimetype || "audio/webm" }), filename || "audio.webm");
  form.append("max_speakers", String(maxSpeakers));
  const baseUrl = (config.transcription.baseUrl || "http://127.0.0.1:8001").replace(/\/$/, "");
  const response = await fetch(`${baseUrl}/speaker-diarization-pyannote`, {
    method: "POST",
    body: form,
    dispatcher: longRunningAgent,
  } as unknown as RequestInit);
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(`Pyannote speaker diarization service failed (${response.status}). ${detail}`);
  }
  const result = await response.json() as { segments?: Segment[]; detectedVoices?: number };
  return { segments: result.segments ?? [], detectedVoices: result.detectedVoices ?? 0 };
}
