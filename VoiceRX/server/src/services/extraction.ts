import { config } from "../config.js";
import type { ExtractResponse } from "../types.js";
import { heuristicExtract } from "./heuristicExtractor.js";
import { bedrockExtract, groqExtract, openaiExtract } from "./llmExtractor.js";

/**
 * Extract structured medical data from a transcript using the configured
 * provider. If an LLM provider fails for any reason, we gracefully fall back to
 * the built-in heuristic extractor and report it via `warning`.
 */
export async function extractMedical(transcript: string): Promise<ExtractResponse> {
  const provider = config.extractionProvider;

  if (provider === "heuristic") {
    return { extraction: heuristicExtract(transcript), provider: "heuristic" };
  }

  try {
    let extraction;
    if (provider === "openai") extraction = await openaiExtract(transcript);
    else if (provider === "groq") extraction = await groqExtract(transcript);
    else extraction = await bedrockExtract(transcript);
    return { extraction, provider };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.warn(`[extract] provider "${provider}" failed, falling back to heuristic: ${message}`);
    return {
      extraction: heuristicExtract(transcript),
      provider: "heuristic",
      warning: `LLM provider "${provider}" failed (${message}). Showing rule-based extraction instead.`,
    };
  }
}
