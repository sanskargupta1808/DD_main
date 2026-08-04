import { config } from "../config.js";

export type SpeakerId = "A" | "B" | "C";
export interface Segment { speaker: SpeakerId; start: number; end: number; text?: string; }
export interface FrequencyGroup {
  speaker: SpeakerId;
  minHz: number;
  maxHz: number;
  meanHz: number;
  samples: number;
}

function splitSentences(text: string): string[] {
  return text.split(/(?<=[.!?।])\s+|\n+/u).map((part) => part.trim()).filter(Boolean);
}

function applySegments(transcript: string, segments: Segment[], duration: number): string {
  const sentences = splitSentences(transcript);
  if (!sentences.length || !segments.length) return transcript;

  // STT frequently returns an unpunctuated paragraph. Splitting only by
  // sentences would then assign the entire transcript to the segment at its
  // midpoint. Fall back to word-level proportional alignment in that case.
  if (sentences.length === 1 && segments.length > 1) {
    const tokens = transcript.trim().split(/\s+/).filter(Boolean);
    const totalChars = tokens.reduce((sum, token) => sum + token.length, 0) || 1;
    const output: string[] = [];
    let used = 0;
    let index = 0;
    let currentSpeaker = "";
    for (const token of tokens) {
      const midpoint = ((used + token.length / 2) / totalChars) * duration;
      while (index + 1 < segments.length && midpoint > segments[index].end) index++;
      const speaker = `Speaker ${segments[index].speaker}`;
      if (speaker !== currentSpeaker) {
        output.push(`${speaker}: ${token}`);
        currentSpeaker = speaker;
      } else {
        output[output.length - 1] += ` ${token}`;
      }
      used += token.length + 1;
    }
    return output.join("\n");
  }

  const totalChars = sentences.reduce((sum, sentence) => sum + sentence.length, 0) || 1;
  const output: string[] = [];
  let used = 0;
  let index = 0;
  for (const sentence of sentences) {
    const midpoint = ((used + sentence.length / 2) / totalChars) * duration;
    while (index + 1 < segments.length && midpoint > segments[index].end) index++;
    const speaker = `Speaker ${segments[index].speaker}`;
    const previous = output[output.length - 1];
    if (previous?.startsWith(`${speaker}:`)) output[output.length - 1] = `${previous} ${sentence}`;
    else output.push(`${speaker}: ${sentence}`);
    used += sentence.length;
  }
  return output.join("\n");
}

export async function frequencyDiarize(
  transcript: string,
  audio: Buffer,
  filename: string,
  mimetype: string,
  maxSpeakers: 2 | 3 = 3,
): Promise<{
  speakerTranscript: string;
  detectedVoices: number;
  segments: Segment[];
  frequencyGroups: FrequencyGroup[];
  speakerChunks: Record<string, string[]>;
}> {
  const form = new FormData();
  form.append("file", new Blob([audio], { type: mimetype || "audio/webm" }), filename || "audio.webm");
  form.append("transcript", transcript);
  form.append("max_speakers", String(maxSpeakers));
  const baseUrl = (config.transcription.baseUrl || "http://127.0.0.1:8001").replace(/\/$/, "");
  const response = await fetch(`${baseUrl}/speaker-diarization`, { method: "POST", body: form });
  if (!response.ok) throw new Error(`Speaker embedding service failed (${response.status}).`);
  const result = await response.json() as {
    segments?: Segment[];
    detectedVoices?: number;
    frequencyGroups?: FrequencyGroup[];
    speakerChunks?: Record<string, string[]>;
  };
  const segments = result.segments ?? [];
  const duration = segments.length ? Math.max(...segments.map((segment) => segment.end)) : 0;
  return {
    speakerTranscript: applySegments(transcript, segments, duration),
    detectedVoices: result.detectedVoices ?? 0,
    segments,
    frequencyGroups: result.frequencyGroups ?? [],
    speakerChunks: result.speakerChunks ?? {},
  };
}
