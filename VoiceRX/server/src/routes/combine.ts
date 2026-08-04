import { Router } from "express";
import { config } from "../config.js";
import { combineRegionalAndEnglish } from "../services/correction.js";

export const combineRouter = Router();

combineRouter.post("/", async (req, res) => {
  const regional = typeof req.body?.regional === "string" ? req.body.regional : "";
  const english = typeof req.body?.english === "string" ? req.body.english : "";
  if (!regional.trim() && !english.trim()) {
    return res.status(400).json({ error: "Provide 'regional' and/or 'english' transcript text." });
  }
  const locale = typeof req.body?.locale === "string" ? req.body.locale : config.correction.locale;
  try {
    const result = await combineRegionalAndEnglish(regional, english, locale);
    res.json(result);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("[combine] failed:", message);
    res.status(500).json({ error: "Combine failed", detail: message });
  }
});
