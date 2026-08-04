import { Router } from "express";
import { config } from "../config.js";
import { correctTranscript } from "../services/correction.js";
import { extractMedical } from "../services/extraction.js";
import { isHighConfidence, searchMedicine } from "../services/medicineSearch.js";
import type { MedicalExtraction, Prescription } from "../types.js";

export const extractRouter = Router();

function normalizeLoose(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim().replace(/\s+/g, " ");
}

function noteMentionsMedicine(note: string, medicine: string): boolean {
  const normalizedNote = normalizeLoose(note);
  const normalizedMedicine = normalizeLoose(medicine);
  if (!normalizedNote || !normalizedMedicine) return false;
  if (normalizedNote === normalizedMedicine) return true;
  return new RegExp(
    `\\b${normalizedMedicine.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`,
    "i"
  ).test(note);
}

async function promoteMedicineMentions(
  extraction: MedicalExtraction,
  medicineTokens: string[]
): Promise<MedicalExtraction> {
  const existing = new Set(extraction.prescriptions.map((p) => normalizeLoose(p.medication)));
  const promoted: Prescription[] = [];
  const remainingNotes: string[] = [];

  for (const note of extraction.notes) {
    let token = medicineTokens.find((medicine) => noteMentionsMedicine(note, medicine));
    if (!token) {
      const medicineResult = await searchMedicine(note);
      if (medicineResult?.match && isHighConfidence(medicineResult)) {
        token = medicineResult.match;
      }
    }
    if (!token) {
      remainingNotes.push(note);
      continue;
    }

    const normalizedToken = normalizeLoose(token);
    if (!existing.has(normalizedToken)) {
      existing.add(normalizedToken);
      promoted.push({ medication: token });
    }

    // Keep the note only if it contains extra information beyond the medicine
    // name itself, so standalone medicine mentions do not get buried in notes.
    if (normalizeLoose(note) !== normalizedToken) {
      remainingNotes.push(note);
    }
  }

  return {
    ...extraction,
    prescriptions: [...extraction.prescriptions, ...promoted],
    notes: remainingNotes,
  };
}

extractRouter.post("/", async (req, res) => {
  const transcript = typeof req.body?.transcript === "string" ? req.body.transcript : "";
  if (!transcript.trim()) {
    return res.status(400).json({ error: "Field 'transcript' (non-empty string) is required." });
  }

  // Per-request override, else fall back to the server-wide setting.
  const doCorrect = typeof req.body?.correct === "boolean" ? req.body.correct : config.correction.enabled;
  const locale = typeof req.body?.locale === "string" ? req.body.locale : config.correction.locale;

  try {
    let textForExtraction = transcript;
    let correctedTranscript: string | undefined;
    let corrections: Record<string, string> | undefined;
    let correctionNote: string | undefined;
    let medicineTokens: string[] = [];

    if (doCorrect) {
      const corr = await correctTranscript(transcript, locale);
      correctedTranscript = corr.corrected;
      corrections = corr.corrections;
      correctionNote = corr.note;
      medicineTokens = corr.medicineTokens;
      if (corr.applied) textForExtraction = corr.corrected;
    }

    const result = await extractMedical(textForExtraction);
    const extraction = doCorrect
      ? await promoteMedicineMentions(result.extraction, medicineTokens)
      : result.extraction;

    res.json({
      ...result,
      extraction,
      correctedTranscript,
      corrections,
      correctionNote,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("[extract] failed:", message);
    res.status(500).json({ error: "Extraction failed", detail: message });
  }
});
