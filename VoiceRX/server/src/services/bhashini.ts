import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { config } from "../config.js";

/**
 * Bhashini / ULCA ASR adapter (Govt of India) — runs AI4Bharat IndicConformer.
 *
 * Two-step flow per the ULCA spec:
 *   1. Config call  → POST {configUrl} with headers userID + ulcaApiKey,
 *      body { pipelineTasks:[{taskType:"asr",config:{language:{sourceLanguage}}}],
 *             pipelineRequestConfig:{pipelineId} }
 *      → returns the ASR serviceId + the Dhruva callbackUrl + an inference
 *        auth header {name,value}.
 *   2. Compute call → POST {callbackUrl} with that auth header, body containing
 *      the base64 WAV audio → returns the transcript.
 *
 * Bhashini wants wav/flac/mp3, so we transcode the browser's webm/m4a to
 * 16 kHz mono WAV with ffmpeg first. Config results are cached per language.
 */

interface ResolvedConfig {
  serviceId: string;
  callbackUrl: string;
  authName: string;
  authValue: string;
}

const _configCache = new Map<string, ResolvedConfig>();

/**
 * Transcode arbitrary audio (webm/m4a/ogg…) to 16 kHz mono WAV via ffmpeg.
 * Uses temp files so ffmpeg writes a correct, seekable WAV header (the pipe
 * form emits streaming placeholder sizes that some parsers reject). Temp files
 * are deleted immediately afterwards (they briefly hold PHI audio on disk).
 */
async function toWav16kMono(input: Buffer, ext: string): Promise<Buffer> {
  const id = randomBytes(8).toString("hex");
  const inPath = join(tmpdir(), `vrx_${id}_in.${ext}`);
  const outPath = join(tmpdir(), `vrx_${id}_out.wav`);
  try {
    await writeFile(inPath, input);
    await new Promise<void>((resolve, reject) => {
      const ff = spawn("ffmpeg", [
        "-hide_banner", "-loglevel", "error", "-y",
        "-i", inPath,
        "-ac", "1",
        "-ar", "16000",
        outPath,
      ]);
      const err: Buffer[] = [];
      ff.stderr.on("data", (d) => err.push(d));
      ff.on("error", (e) => reject(new Error(`ffmpeg not available (install ffmpeg): ${e.message}`)));
      ff.on("close", (code) =>
        code === 0
          ? resolve()
          : reject(new Error(`ffmpeg transcode failed: ${Buffer.concat(err).toString().slice(0, 200)}`))
      );
    });
    return await readFile(outPath);
  } finally {
    await rm(inPath, { force: true }).catch(() => {});
    await rm(outPath, { force: true }).catch(() => {});
  }
}

async function resolvePipeline(lang: string): Promise<ResolvedConfig> {
  const cached = _configCache.get(lang);
  if (cached) return cached;

  if (!config.bhashini.userId || !config.bhashini.ulcaApiKey) {
    throw new Error("BHASHINI_USER_ID and BHASHINI_ULCA_API_KEY must be set.");
  }

  const res = await fetch(config.bhashini.configUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      userID: config.bhashini.userId,
      ulcaApiKey: config.bhashini.ulcaApiKey,
    },
    body: JSON.stringify({
      pipelineTasks: [{ taskType: "asr", config: { language: { sourceLanguage: lang } } }],
      pipelineRequestConfig: { pipelineId: config.bhashini.pipelineId },
    }),
  });
  if (!res.ok) {
    throw new Error(`Bhashini config call failed ${res.status}: ${(await res.text()).slice(0, 200)}`);
  }
  const data: any = await res.json();

  const asrTask = (data?.pipelineResponseConfig ?? []).find((t: any) => t.taskType === "asr");
  const serviceId: string | undefined = asrTask?.config?.[0]?.serviceId;
  const endpoint = data?.pipelineInferenceAPIEndPoint;
  const callbackUrl: string | undefined = endpoint?.callbackUrl;
  const authName: string | undefined = endpoint?.inferenceApiKey?.name;
  const authValue: string | undefined = endpoint?.inferenceApiKey?.value;

  if (!serviceId || !callbackUrl || !authName || !authValue) {
    throw new Error(`Bhashini config response missing ASR serviceId/endpoint for "${lang}".`);
  }

  const resolved: ResolvedConfig = { serviceId, callbackUrl, authName, authValue };
  _configCache.set(lang, resolved);
  return resolved;
}

export async function bhashiniTranscribe(
  buffer: Buffer,
  mimetype: string,
  langOrLocale: string
): Promise<{ transcript: string; model: string; provider: string; language: string }> {
  // Bhashini ASR requires a source language; default to Hindi if "auto"/unknown.
  const lang = (langOrLocale || "hi").slice(0, 2).toLowerCase() || "hi";

  const ext = mimetype.includes("mp4") || mimetype.includes("m4a")
    ? "m4a"
    : mimetype.includes("ogg")
      ? "ogg"
      : mimetype.includes("wav")
        ? "wav"
        : mimetype.includes("mpeg") || mimetype.includes("mp3")
          ? "mp3"
          : "webm";

  const pipeline = await resolvePipeline(lang);
  const wav = await toWav16kMono(buffer, ext);
  const audioContent = wav.toString("base64");

  const res = await fetch(pipeline.callbackUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      [pipeline.authName]: pipeline.authValue,
    },
    body: JSON.stringify({
      pipelineTasks: [
        {
          taskType: "asr",
          config: {
            language: { sourceLanguage: lang },
            serviceId: pipeline.serviceId,
            audioFormat: "wav",
            samplingRate: 16000,
            preProcessors: ["vad"],
            postProcessors: ["itn", "punctuation"],
          },
        },
      ],
      inputData: {
        input: [{ source: null }],
        audio: [{ audioContent }],
      },
    }),
  });

  if (!res.ok) {
    throw new Error(`Bhashini compute call failed ${res.status}: ${(await res.text()).slice(0, 200)}`);
  }
  const data: any = await res.json();
  const transcript: string =
    data?.pipelineResponse?.[0]?.output?.[0]?.source ?? "";

  return { transcript: transcript.trim(), model: pipeline.serviceId, provider: "bhashini", language: lang };
}
