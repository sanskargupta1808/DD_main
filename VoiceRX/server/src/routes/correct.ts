import { Router } from "express";
import { config } from "../config.js";
import { correctTranscript } from "../services/correction.js";

export const correctRouter = Router();

correctRouter.post("/", async (req, res) => {
  const transcript = typeof req.body?.transcript === "string" ? req.body.transcript : "";
  if (!transcript.trim()) {
    return res.status(400).json({ error: "Field 'transcript' (non-empty string) is required." });
  }
  const locale = typeof req.body?.locale === "string" ? req.body.locale : config.correction.locale;
  try {
    const result = await correctTranscript(transcript, locale);
    res.json(result);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("[correct] failed:", message);
    res.status(500).json({ error: "Correction failed", detail: message });
  }
});
