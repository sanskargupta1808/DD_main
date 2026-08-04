import { config } from "../config.js";
import { correctLabeledTranscript, groqChat } from "./correction.js";
import { frequencyDiarize, type FrequencyGroup, type Segment, type SpeakerId } from "./frequencyDiarize.js";
import { pyannoteDiarize } from "./pyannoteDiarize.js";
import {
  transcribeEnglishForced,
  transcribeWithMode,
  transcribeWithWordTimestamps,
  type WordTimestamp,
} from "./transcription.js";
import { createSpeakerTracks } from "./speakerAudio.js";

/**
 * Label a doctor–patient consultation transcript by speaker role (content-based,
 * via the LLM — not acoustic voiceprint). Groups consecutive same-speaker
 * sentences into one turn and preserves the exact wording/language.
 */
const SYSTEM_PROMPT = `You label a doctor–patient consultation transcript by speaker.

Rules:
- Split the transcript into turns and prefix each turn with "Doctor:" or "Patient:".
- Group CONSECUTIVE sentences spoken by the same person into a SINGLE turn.
- Infer the speaker from context: the DOCTOR greets, asks history, examines, states
  vitals/findings, diagnoses, prescribes medicines, and gives advice; the PATIENT
  describes symptoms, history, and answers the doctor's questions.
- Do NOT translate, add, remove, or reword anything — keep the exact text and the
  original language/script. Only add the "Doctor:"/"Patient:" prefixes.
- Output ONLY the labeled transcript: one turn per line, e.g.
  "Doctor: ..."
  "Patient: ..."`;

export async function diarizeTranscript(
  transcript: string,
  _locale: string = config.correction.locale
): Promise<string> {
  if (!transcript.trim() || !config.groq.apiKey) return transcript;
  try {
    const maxTokens = Math.min(8000, Math.max(2048, Math.ceil(transcript.length * 2)));
    const labeled = await groqChat(
      [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: transcript },
      ],
      0.1,
      maxTokens,
      config.groq.extractionModel
    );
    return labeled.trim() || transcript;
  } catch (err) {
    console.warn("[diarize] failed:", err instanceof Error ? err.message : err);
    return transcript;
  }
}

const HYBRID_PROMPT = `You receive a transcript whose turns are labeled Speaker A or Speaker B.
Determine which speaker is the doctor and which is the patient from the conversation context.
Replace ONLY the Speaker A/Speaker B prefixes with Doctor/Patient. Preserve every word,
language, order, and line break. Output only the labeled transcript.`;

export async function hybridDiarize(
  transcript: string,
  audio: Buffer,
  filename: string,
  mimetype: string,
  locale: string = config.correction.locale,
  maxSpeakers: 2 | 3 = 3,
): Promise<{
  diarized: string;
  detectedVoices: number;
  segments: Array<{ speaker: SpeakerId; start: number; end: number }>;
  frequencyGroups: FrequencyGroup[];
  speakerChunks: Record<string, string[]>;
  speakerContainers: Array<{
    speaker: SpeakerId;
    label: string;
    chunks: string[];
    transcript: string;
  }>;
}> {
  const acoustic = await frequencyDiarize(transcript, audio, filename, mimetype, maxSpeakers);
  const speakerContainers = await transcribeSpeakerContainers(
    acoustic.speakerChunks,
    acoustic.segments,
    audio,
    filename,
    mimetype,
    locale,
  );
  const independentChunks = Object.fromEntries(
    speakerContainers.map((container) => [container.speaker, container.chunks])
  );
  if (!config.groq.apiKey || acoustic.detectedVoices < 2) {
    return {
      diarized: acoustic.speakerTranscript,
      detectedVoices: acoustic.detectedVoices,
      segments: acoustic.segments,
      frequencyGroups: acoustic.frequencyGroups,
      speakerChunks: independentChunks,
      speakerContainers,
    };
  }
  try {
    const labeled = await groqChat(
      [{ role: "system", content: HYBRID_PROMPT }, { role: "user", content: acoustic.speakerTranscript }],
      0,
      Math.min(8000, Math.max(2048, Math.ceil(transcript.length * 2))),
      config.groq.extractionModel
    );
    return {
      diarized: labeled || acoustic.speakerTranscript,
      detectedVoices: acoustic.detectedVoices,
      segments: acoustic.segments,
      frequencyGroups: acoustic.frequencyGroups,
      speakerChunks: independentChunks,
      speakerContainers,
    };
  } catch {
    return {
      diarized: acoustic.speakerTranscript,
      detectedVoices: acoustic.detectedVoices,
      segments: acoustic.segments,
      frequencyGroups: acoustic.frequencyGroups,
      speakerChunks: independentChunks,
      speakerContainers,
    };
  }
}

/**
 * Assign each transcribed word to whichever acoustic speaker segment is
 * active at that word's timestamp, then group consecutive same-speaker
 * words into turns. This is real alignment, not a guess: it only works
 * because the transcription came from a single whole-recording pass, so
 * every word carries a genuine timestamp from that pass.
 */
function attributeWordsToSpeakers(
  words: WordTimestamp[],
  segments: Segment[],
): Array<{ speaker: SpeakerId; user: string; start: number; end: number; text: string }> {
  if (!words.length || !segments.length) return [];
  const sorted = [...segments].sort((a, b) => a.start - b.start);
  const userNumber = new Map<SpeakerId, number>();
  const turns: Array<{ speaker: SpeakerId; user: string; start: number; end: number; text: string }> = [];

  let segIndex = 0;
  for (const word of words) {
    const midpoint = (word.start + word.end) / 2;
    while (segIndex + 1 < sorted.length && midpoint >= sorted[segIndex].end) segIndex += 1;
    const speaker = sorted[segIndex].speaker;
    if (!userNumber.has(speaker)) userNumber.set(speaker, userNumber.size + 1);
    const user = `User ${userNumber.get(speaker)}`;

    const last = turns[turns.length - 1];
    if (last && last.speaker === speaker) {
      last.text = `${last.text} ${word.word}`;
      last.end = word.end;
    } else {
      turns.push({ speaker, user, start: word.start, end: word.end, text: word.word });
    }
  }
  return turns;
}

/**
 * Real acoustic diarization (pyannote) + real word-level timestamps from a
 * single whole-recording Groq Whisper pass, aligned against each other, then
 * a text-level correction pass (medicine-name NER + FAISS "our model" lookup,
 * plus a general code-switch script fix using a second English-forced
 * transcript as a reference) applied to the already-labeled transcript.
 *
 * Deliberately LABEL FIRST, CORRECT SECOND — not the other way round:
 *   - Zero isolated-clip transcription: handed a bare 1-2s clip of a quick
 *     back-and-forth exchange with no surrounding context, Whisper (and
 *     IndicConformer) reliably hallucinate fluent nonsense ("subscribe to
 *     our channel"). Transcribing the whole recording once has the context
 *     to be reliable, and real per-word timestamps let us attribute speakers
 *     without any proportional guessing.
 *   - Correction/combine steps are LLM text-rewrites that don't preserve
 *     word timestamps. Running correction on the already-labeled transcript
 *     (correctLabeledTranscript, explicitly told to preserve the "User N:"
 *     line structure) keeps the real timestamp-based labels intact, instead
 *     of needing to re-derive them from a rewritten transcript afterwards.
 *
 * Speakers are labeled generically "User 1"/"User 2" in first-appearance
 * order (not guessed as Doctor/Patient), and no LLM is involved in deciding
 * who spoke when — only in cleaning up what was said.
 */
function buildAcousticSpeakerContainers(
  turns: Array<{ speaker: SpeakerId; user: string; start: number; end: number; text: string }>
): Array<{ speaker: SpeakerId; label: string; chunks: string[]; transcript: string }> {
  const bySpeaker = new Map<SpeakerId, { label: string; chunks: string[] }>();
  for (const turn of turns) {
    const entry = bySpeaker.get(turn.speaker) ?? { label: turn.user, chunks: [] };
    entry.chunks.push(turn.text);
    bySpeaker.set(turn.speaker, entry);
  }
  return Array.from(bySpeaker.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([speaker, { label, chunks }]) => ({ speaker, label, chunks, transcript: chunks.join(" ") }));
}

export async function acousticDiarize(
  audio: Buffer,
  filename: string,
  mimetype: string,
  locale: string = config.correction.locale,
  maxSpeakers: 2 | 3 = 3,
): Promise<{
  diarized: string;
  detectedVoices: number;
  segments: Segment[];
  turns: Array<{ speaker: SpeakerId; user: string; start: number; end: number; text: string }>;
  speakerContainers: Array<{ speaker: SpeakerId; label: string; chunks: string[]; transcript: string }>;
  independentTranscription: boolean;
  corrections: Record<string, string>;
  medicineTokens: string[];
}> {
  // Parallel: pyannote (local indic_stt) and the two Groq Whisper calls are
  // independent calls on the same buffer. An earlier "fetch failed" here
  // turned out to be indic_stt degrading under sustained load and exceeding
  // undici's default 300s timeout (see pyannoteDiarize's longRunningAgent),
  // not real contention from running these at once — confirmed by that same
  // timeout recurring later even when they were sequential. Running them
  // concurrently is safe now that the real cause has its own fix.
  //
  // englishReference (transcribeEnglishForced) exists specifically to fix
  // code-switching generally, not just medicine names: a prompt-only attempt
  // without it either under-corrected (no reference to check against) or, when
  // given a long example list to compensate, broke the "output only" format
  // constraint. Giving the cleanup pass an actual second transcript to check
  // against fixes this reliably without either failure mode.
  const [{ segments, detectedVoices }, transcription, englishReference] = await Promise.all([
    pyannoteDiarize(audio, filename, mimetype, maxSpeakers),
    transcribeWithWordTimestamps(audio, filename, mimetype, locale),
    transcribeEnglishForced(audio, filename, mimetype).catch((err) => {
      console.warn("[acousticDiarize] English reference transcription failed:", err instanceof Error ? err.message : err);
      return "";
    }),
  ]);

  if (detectedVoices < 2 || !segments.length || !transcription.words.length) {
    const text = transcription.text.trim();
    if (!text) {
      return {
        diarized: "",
        detectedVoices,
        segments,
        turns: [],
        speakerContainers: [],
        independentTranscription: true,
        corrections: {},
        medicineTokens: [],
      };
    }
    // Label first, correct second — see correctLabeledTranscript's docstring.
    const result = await correctLabeledTranscript(`User 1: ${text}`, locale, englishReference);
    const singleTurn = [{ speaker: "A" as SpeakerId, user: "User 1", start: 0, end: 0, text }];
    return {
      diarized: result.corrected,
      detectedVoices,
      segments,
      turns: singleTurn,
      speakerContainers: buildAcousticSpeakerContainers(singleTurn),
      independentTranscription: true,
      corrections: result.corrections,
      medicineTokens: result.medicineTokens,
    };
  }

  const turns = attributeWordsToSpeakers(transcription.words, segments);
  const rawLabeled = turns.map((t) => `${t.user}: ${t.text}`).join("\n");
  // Label first (real timestamps), correct second (text-level medicine-name
  // fixup via NER + FAISS + Groq cleanup, preserving the User N: structure) —
  // see acousticDiarize's docstring for why this order, not the reverse.
  const result = await correctLabeledTranscript(rawLabeled, locale, englishReference);
  return {
    diarized: result.corrected,
    detectedVoices,
    segments,
    turns,
    speakerContainers: buildAcousticSpeakerContainers(turns),
    independentTranscription: true,
    corrections: result.corrections,
    medicineTokens: result.medicineTokens,
  };
}

function buildSpeakerContainers(
  speakerChunks: Record<string, string[]>
): Array<{ speaker: SpeakerId; label: string; chunks: string[]; transcript: string }> {
  return Object.entries(speakerChunks)
    .filter(([speaker]) => speaker === "A" || speaker === "B" || speaker === "C")
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([speaker, chunks]) => ({
      speaker: speaker as SpeakerId,
      label: `Speaker ${speaker}`,
      chunks,
      transcript: chunks.join(" "),
    }));
}

async function transcribeSpeakerContainers(
  fallbackChunks: Record<string, string[]>,
  segments: Array<{ speaker: SpeakerId; start: number; end: number }>,
  audio: Buffer,
  filename: string,
  mimetype: string,
  locale: string,
): Promise<Array<{ speaker: SpeakerId; label: string; chunks: string[]; transcript: string }>> {
  let tracks: Map<SpeakerId, Buffer>;
  try {
    tracks = await createSpeakerTracks(audio, filename, mimetype, segments);
  } catch (err) {
    console.warn("[diarize] speaker track extraction failed:", err instanceof Error ? err.message : err);
    return buildSpeakerContainers(fallbackChunks);
  }

  const speakers = Array.from(new Set([
    ...Array.from(tracks.keys()),
    ...Object.keys(fallbackChunks).filter((key): key is SpeakerId => key === "A" || key === "B" || key === "C"),
  ])).sort();

  const containers = await Promise.all(speakers.map(async (speaker) => {
    const track = tracks.get(speaker);
    if (!track) {
      const chunks = fallbackChunks[speaker] ?? [];
      return { speaker, label: `Speaker ${speaker}`, chunks, transcript: chunks.join(" ") };
    }
    try {
      const result = await transcribeWithMode(
        track,
        `speaker-${speaker}.wav`,
        "audio/wav",
        locale,
        "auto",
      );
      const text = result.transcript.trim();
      const chunks = splitTranscriptChunks(text || (fallbackChunks[speaker] ?? []).join(" "));
      return { speaker, label: `Speaker ${speaker}`, chunks, transcript: chunks.join(" ") };
    } catch (err) {
      console.warn(`[diarize] speaker ${speaker} transcription failed:`, err instanceof Error ? err.message : err);
      const chunks = fallbackChunks[speaker] ?? [];
      return { speaker, label: `Speaker ${speaker}`, chunks, transcript: chunks.join(" ") };
    }
  }));
  return containers;
}

function splitTranscriptChunks(text: string): string[] {
  return text
    .split(/(?<=[.!?।])\s+|\n+/u)
    .map((chunk) => chunk.trim())
    .filter(Boolean);
}
