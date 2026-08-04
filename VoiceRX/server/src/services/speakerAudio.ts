import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import type { Segment, SpeakerId } from "./frequencyDiarize.js";

const execFileAsync = promisify(execFile);

/**
 * Build one concatenated, single-speaker WAV for every acoustic cluster.
 * The diarizer gives us time ranges; this function removes the other voices
 * by keeping only that cluster's ranges, then concatenates those ranges in
 * chronological order for independent transcription.
 */
export async function createSpeakerTracks(
  audio: Buffer,
  filename: string,
  mimetype: string,
  segments: Segment[]
): Promise<Map<SpeakerId, Buffer>> {
  const groups = new Map<SpeakerId, Segment[]>();
  for (const segment of segments) {
    if (segment.end <= segment.start || segment.end - segment.start < 0.25) continue;
    const list = groups.get(segment.speaker) ?? [];
    list.push(segment);
    groups.set(segment.speaker, list);
  }
  if (!groups.size) return new Map();

  const work = await mkdtemp(join(tmpdir(), "voicerx-speakers-"));
  const input = join(work, `input${extensionFor(filename, mimetype)}`);
  const tracks = new Map<SpeakerId, Buffer>();
  try {
    await writeFile(input, audio);
    for (const [speaker, speakerSegments] of groups) {
      const parts: string[] = [];
      for (let index = 0; index < speakerSegments.length; index += 1) {
        const segment = speakerSegments[index];
        const part = join(work, `${speaker}-${index}.wav`);
        const duration = Math.max(0.25, segment.end - segment.start);
        await execFileAsync("ffmpeg", [
          "-hide_banner", "-loglevel", "error", "-y",
          "-ss", String(Math.max(0, segment.start)),
          "-t", String(duration),
          "-i", input,
          "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", part,
        ]);
        parts.push(part);
      }
      if (!parts.length) continue;

      const listFile = join(work, `${speaker}-concat.txt`);
      const output = join(work, `${speaker}.wav`);
      const concatList = parts
        .map((part) => `file '${part.replaceAll("'", "'\\''")}'`)
        .join("\n");
      await writeFile(listFile, `${concatList}\n`, "utf8");
      await execFileAsync("ffmpeg", [
        "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", listFile,
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", output,
      ]);
      tracks.set(speaker, await readFile(output));
    }
    return tracks;
  } finally {
    await rm(work, { recursive: true, force: true });
  }
}

function extensionFor(filename: string, mimetype: string): string {
  const match = filename.match(/\.[a-z0-9]+$/i)?.[0];
  if (match) return match;
  if (mimetype.includes("mp4") || mimetype.includes("m4a")) return ".m4a";
  if (mimetype.includes("ogg")) return ".ogg";
  if (mimetype.includes("wav")) return ".wav";
  return ".webm";
}
