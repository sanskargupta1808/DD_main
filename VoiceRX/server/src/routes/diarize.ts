import { Router } from "express";
import multer from "multer";
import { config } from "../config.js";
import { acousticDiarize, diarizeTranscript, hybridDiarize } from "../services/diarize.js";

export const diarizeRouter = Router();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 25 * 1024 * 1024 } });

diarizeRouter.post("/", upload.single("audio"), async (req, res) => {
  const mode = req.body?.mode === "hybrid" ? "hybrid" : req.body?.mode === "acoustic" ? "acoustic" : "ai";
  const transcript = typeof req.body?.transcript === "string" ? req.body.transcript : "";
  if (mode !== "acoustic" && !transcript.trim()) {
    return res.status(400).json({ error: "Field 'transcript' (non-empty string) is required." });
  }
  const locale = typeof req.body?.locale === "string" ? req.body.locale : config.correction.locale;
  const maxSpeakers = req.body?.max_speakers === "2" ? 2 : 3;
  try {
    if (mode === "acoustic") {
      if (!req.file) return res.status(400).json({ error: "Audio is required for acoustic diarization." });
      const result = await acousticDiarize(
        req.file.buffer,
        req.file.originalname,
        req.file.mimetype,
        locale,
        maxSpeakers,
      );
      return res.json(result);
    }
    if (mode === "hybrid") {
      if (!req.file) return res.status(400).json({ error: "Audio is required for hybrid diarization." });
      const result = await hybridDiarize(
        transcript,
        req.file.buffer,
        req.file.originalname,
        req.file.mimetype,
        locale,
        maxSpeakers,
      );
      return res.json(result);
    }
    const diarized = await diarizeTranscript(transcript, locale);
    res.json({ diarized });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("[diarize] failed:", err);
    res.status(500).json({ error: "Diarization failed", detail: message });
  }
});
