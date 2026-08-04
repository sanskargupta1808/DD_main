import { Router } from "express";
import multer from "multer";
import { type TranscribeMode, transcribeWithMode } from "../services/transcription.js";

// Audio stays in memory only — never written to disk (PHI safety).
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 }, // 25 MB
});

export const transcribeRouter = Router();

transcribeRouter.post("/", upload.single("audio"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "An audio file (field 'audio') is required." });
  }
  const locale = typeof req.body?.locale === "string" ? req.body.locale : undefined;
  const modeRaw = typeof req.body?.mode === "string" ? req.body.mode : "auto";
  const mode: TranscribeMode =
    modeRaw === "regional" || modeRaw === "english" ? modeRaw : "auto";
  try {
    const result = await transcribeWithMode(
      req.file.buffer,
      req.file.originalname,
      req.file.mimetype,
      locale,
      mode
    );
    res.json(result);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("[transcribe] failed:", message);
    res.status(500).json({ error: "Transcription failed", detail: message });
  }
});
