import { config } from "../config.js";
import { isHighConfidence, searchMedicine } from "./medicineSearch.js";

/**
 * STT transcript correction — TypeScript port of
 * lib/global/services/speech_ai/groq_transcript_processor.dart.
 *
 * Pipeline:
 *   0. Alias pre-scan      — hardcoded phonetic STT hallucinations.
 *   1. Groq medicine NER   — extract medicine tokens from the transcript.
 *   2. ML normalization    — resolve each token via the FAISS medicine service.
 *   3. Groq final cleanup  — Indian-OPD prompt with known corrections injected.
 *
 * Every stage degrades gracefully: missing Groq key, unreachable medicine
 * service, or any API error falls back to the best transcript produced so far,
 * so the extraction flow is never broken. The Groq key is read from server env
 * (GROQ_API_KEY) — the browser never sees it.
 */

/** Hardcoded phonetic STT hallucinations no ML engine can recover. */
const ALIASES: Record<string, string> = {
  "cal coral dee": "Kalcoral D",
  "calculate d": "Kalcoral D",
  "kal coral d": "Kalcoral D",
  "karakural d": "Kalcoral D",
  karakural: "Kalcoral",
  "ravi prasoon": "Rabeprazole",
  "ravi prasual": "Rabeprazole",
  "ravi prasal": "Rabeprazole",
  "ravi prasul": "Rabeprazole",
  "ravi prazool": "Rabeprazole",
  "happy prasul": "Rabeprazole",
  "happy prasoon": "Rabeprazole",
  "happy prasal": "Rabeprazole",
  "happi prasul": "Rabeprazole",
  "happi prasoon": "Rabeprazole",
  "brazil brazil": "Rabeprazole",
  "baby brazil": "Rabeprazole",
  "baby prasoon": "Rabeprazole",
  "baby prasul": "Rabeprazole",
  "baby prasal": "Rabeprazole",
  enterprises: "Pantoprazole",
};

/**
 * Common English words Indian speakers code-switch into mid-sentence, which
 * the STT engine renders phonetically in Devanagari instead of Latin script.
 * Applied as a deterministic regex replace — not an LLM judgment call — because
 * three different LLM-based approaches (a long example list, a short one, and
 * a second reference transcript to cross-check against) all proved unreliable
 * at catching this consistently. This won't catch every possible code-switched
 * word, only the common ones listed here, but what it does catch, it catches
 * every time. Multiple spellings per word because Whisper's phonetic rendering
 * of the same word varies between runs (e.g. "hallucination" alone showed up
 * as एलूशिनेशन, हेलुसिनेशन, हैलुसिनेशन, and हेल्यूसिनेशन across test runs).
 */
const CODE_SWITCH_ALIASES: Record<string, string> = {
  डॉक्टर: "doctor",
  डाक्टर: "doctor",
  हॉस्पिटल: "hospital",
  हास्पिटल: "hospital",
  रिसेप्शन: "reception",
  क्लिनिक: "clinic",
  फोकस: "focus",
  मोबाइल: "mobile",
  लैपटॉप: "laptop",
  स्क्रीन: "screen",
  इंजीनियर: "engineer",
  इमरजेंसी: "emergency",
  इमरजनसी: "emergency",
  सजेशन: "suggestion",
  चेकअप: "checkup",
  एक्टिव: "active",
  बिज़ी: "busy",
  बिजी: "busy",
  हेलुसिनेशन: "hallucination",
  हैलुसिनेशन: "hallucination",
  "हेल्यूसिनेशन": "hallucination",
  एलूशिनेशन: "hallucination",
  हलुसिनेशन: "hallucination",
  हेलुसिनेट: "hallucinate",
  परासिटमोल: "Paracetamol",
  "पैरासिटामोल": "Paracetamol",
};

export interface CorrectionResult {
  /** The corrected transcript (or the original if correction was skipped). */
  corrected: string;
  /** Map of original phrase/token → corrected name that was applied. */
  corrections: Record<string, string>;
  /** Medicine tokens detected by Groq NER. */
  medicineTokens: string[];
  /** Whether any correction stage actually ran / changed the text. */
  applied: boolean;
  /** Human-readable note when correction was degraded or skipped. */
  note?: string;
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function languageHint(localeId: string): string {
  switch (localeId) {
    case "hi_IN":
      return "in Hindi or a mix of Hindi and English (Hinglish)";
    case "gu_IN":
      return "in Gujarati or a mix of Gujarati and English";
    case "mr_IN":
      return "in Marathi or a mix of Marathi and English";
    case "bn_IN":
      return "in Bengali or a mix of Bengali and English";
    case "ta_IN":
      return "in Tamil or a mix of Tamil and English";
    case "te_IN":
      return "in Telugu or a mix of Telugu and English";
    case "kn_IN":
      return "in Kannada or a mix of Kannada and English";
    case "ml_IN":
      return "in Malayalam or a mix of Malayalam and English";
    case "pa_IN":
      return "in Punjabi or a mix of Punjabi and English";
    case "or_IN":
      return "in Odia or a mix of Odia and English";
    default:
      return "in English (Indian accent / medical terminology)";
  }
}

function buildSystemPrompt(localeId: string, corrections: Record<string, string>): string {
  const correctionHint =
    Object.keys(corrections).length === 0
      ? ""
      : "\n\nKnown medicine corrections (apply these exactly):\n" +
        Object.entries(corrections)
          .map(([k, v]) => `  "${k}" → "${v}"`)
          .join("\n");

  return `You are a medical transcription corrector specialised for Indian OPD (outpatient department) dictations.
The doctor dictates patient notes ${languageHint(localeId)}.
Your ONLY task is to correct errors introduced by the speech-to-text engine and return the cleaned transcript.

Rules:
1. Return ONLY the corrected transcript – no explanations, no labels, no extra text. NEVER shorten, summarise, or drop any part of what was spoken. Every word the doctor said must appear in the output.
2. Do NOT add, remove, or infer ANY information that was not spoken. Never suggest medicines, diagnoses, symptoms, or advice that the doctor did not explicitly dictate. If the doctor says "patient has fever", do NOT add any medicine names.
3. Do NOT translate between languages. Preserve the original language(s) as spoken.
4. Fix Indian brand/generic drug name recognition errors ONLY when the doctor actually spoke a medicine name that was misheard. Common drugs include:
   Paracetamol, Calpol 650, Dolo 650, Combiflam, Kalcoral D (often misheard as 'calculate d' or 'cal coral dee'),
   Augmentin, Azithromycin, Amoxicillin, Pantoprazole, Pan-D, Pantop,
   Zerodol-SP, Zerodol, Crocin, Meftal-Spas, Allegra, Montek LC,
   Voveran, Tramadol, Deriphyllin, Ascoril, Benadryl, Omnacortil,
   Metformin, Glimepiride, Telma-H, Amlodipine, Atorvastatin, Ecosprin,
   Ranitidine, Omez, Rabeprazole, Ondansetron, Domstal, Emeset,
   Gentamicin, Clindamycin, Doxycycline, Ciprofloxacin, Norfloxacin.
5. Normalise vitals phrasing:
   - "temperature one oh one" → "temperature 101"
   - "bp one twenty by eighty" → "BP 120/80"
   - "pulse seventy two" → "pulse 72"
   - "spo2 ninety eight percent" → "SpO2 98%"
   - "height one seventy centimeters" → "height 170 cm"
   - "weight seventy kilograms" → "weight 70 kg"
6. Fix frequency / dosage phrasing:
   - "six fifty mg" → "650 mg"
   - "twice a day" / "twice daily" / "BD" are equivalent – keep as spoken.
   - "once daily" / "OD", "thrice daily" / "TDS" – keep as spoken.
   - "IMP" / "imp" before a medicine = "immediately" (STAT dose) — fix to "immediately".
     e.g. "give IMP Paracetamol" → "give Paracetamol immediately"
   - "STAT" = immediately, keep as spoken or expand if context is clearer.
7. Keep clinical values (numbers, units) accurate – never guess or alter clinical meaning.
8. If the transcript is already clean, return it unchanged.
9. ALWAYS write medicine/drug names in English (Latin script) in their standard
   spelling (e.g. "Paracetamol", "Dolo 650", "Pantoprazole"), even when the rest of
   the dictation is in Hindi or another Indian language/script. Keep the EXACT drug
   spoken — do NOT swap a brand name for its generic or vice versa (if "Dolo" was
   said, keep "Dolo"). Do NOT translate or transliterate any non-medicine words —
   keep all other content in the original language exactly as spoken.${correctionHint}
`;
}

interface GroqMessage {
  role: "system" | "user";
  content: string;
}

export async function groqChat(
  messages: GroqMessage[],
  temperature: number,
  maxTokens: number,
  model: string = config.groq.model
): Promise<string> {
  const res = await fetch(`${config.groq.baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${config.groq.apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages,
      temperature,
      max_tokens: maxTokens,
    }),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Groq API error ${res.status}: ${body.slice(0, 200)}`);
  }
  const data: any = await res.json();
  const content = data?.choices?.[0]?.message?.content;
  if (typeof content !== "string") throw new Error("Groq returned no content");
  return content.trim();
}

/** Step 1: Groq medical NER → list of medicine tokens. */
async function extractMedicineTokens(text: string): Promise<string[]> {
  try {
    const content = await groqChat(
      [
        {
          role: "system",
          content:
            "You are a medical NER system. Extract ONLY medicine/drug names " +
            "from the transcript — including brand names, generic names, and " +
            "any misspelled/misheard medicine names. " +
            "Return a JSON array of strings, nothing else. " +
            'Example: ["Paracetamol", "Tachosil Mini 3.0cm x 2.5cm patch", "Rabipra 650"] ' +
            "If no medicines found, return [].",
        },
        { role: "user", content: text },
      ],
      0,
      512
    );
    const match = content.match(/\[[\s\S]*?\]/);
    if (!match) return [];
    const parsed = JSON.parse(match[0]);
    return Array.isArray(parsed) ? parsed.filter((x) => typeof x === "string") : [];
  } catch (err) {
    console.warn("[correction] medicine NER failed:", err instanceof Error ? err.message : err);
    return [];
  }
}

/** Strength tokens like "500mg", "20 mg", "5ml", "100 iu". */
const STRENGTH_RE = /\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml|g|iu|units?|%)\b/gi;
/** Dosage-form / pack words that should never be added to a spoken name. */
const FORM_RE =
  /\b(?:tablets?|tab|capsules?|cap|syrup|suspension|solution|injections?|inj|infusion|drops?|gel|cream|ointment|lotion|spray|inhaler|patch|sachet|powder|granules?|suppository|respules?|kit)\b/gi;

/** Brand_name with strength/form stripped, original casing preserved. */
function strippedBrand(s: string): string {
  return s.replace(STRENGTH_RE, " ").replace(FORM_RE, " ").replace(/\s+/g, " ").trim();
}

/** Lowercased word list of a name with strength/form/punctuation removed. */
function coreWords(s: string): string[] {
  const t = strippedBrand(s)
    .toLowerCase()
    .replace(/[^a-z0-9+\s-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return t ? t.split(" ") : [];
}

/** Normalised Levenshtein similarity (0..1). */
function similarity(a: string, b: string): number {
  if (a === b) return 1;
  if (!a || !b) return 0;
  const m = a.length;
  const n = b.length;
  let prev = Array.from({ length: n + 1 }, (_, i) => i);
  for (let i = 1; i <= m; i++) {
    const curr = [i, ...Array(n).fill(0)];
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost);
    }
    prev = curr;
  }
  return 1 - prev[n] / Math.max(m, n);
}

/**
 * Accept a medicine-search result ONLY when it corrects the spelling of the
 * same spoken medicine name. Rejects brand/strength/form expansions
 * (e.g. "Rabeprazole" → "Roseate Rabeprazole 20mg Tablet") and unrelated
 * brand substitutions. Returns the cleaned name, or null to leave it unchanged.
 */
function spellingFixOnly(token: string, matchBrand: string): string | null {
  const tokWords = coreWords(token);
  const matchWords = coreWords(matchBrand);
  if (tokWords.length === 0 || matchWords.length === 0) return null;

  // Adding words = adding a brand/strength/form the doctor never said → reject.
  if (matchWords.length > tokWords.length) return null;

  // Must be the same name with only spelling differences.
  if (similarity(tokWords.join(" "), matchWords.join(" ")) < 0.7) return null;

  const cleaned = strippedBrand(matchBrand);
  // No-op if it's identical (ignoring case) to what was already spoken.
  return cleaned.toLowerCase() === token.toLowerCase().trim() ? null : cleaned;
}

/** Step 2: resolve each token via alias table, then the FAISS medicine service. */
async function normalizeMedicineTokens(tokens: string[]): Promise<Record<string, string>> {
  const corrections: Record<string, string> = {};
  for (const token of tokens) {
    const normalized = token.toLowerCase().trim();

    if (ALIASES[normalized]) {
      corrections[token] = ALIASES[normalized];
      continue;
    }

    const result = await searchMedicine(token);
    if (result && result.match && isHighConfidence(result)) {
      const fix = spellingFixOnly(token, result.match);
      if (fix) corrections[token] = fix;
    }
  }
  return corrections;
}

/** Apply a corrections map by literal, case-insensitive replacement. */
function applyCorrections(text: string, corrections: Record<string, string>): string {
  let out = text;
  for (const [from, to] of Object.entries(corrections)) {
    out = out.replace(new RegExp(escapeRegExp(from), "gi"), to);
  }
  return out;
}

/**
 * Correct a raw STT transcript. Mirrors GroqTranscriptProcessor.process().
 * Never throws — returns the best transcript it can produce.
 */
export async function correctTranscript(
  rawText: string,
  localeId: string = config.correction.locale
): Promise<CorrectionResult> {
  const corrections: Record<string, string> = {};
  if (!rawText.trim()) {
    return { corrected: rawText, corrections, medicineTokens: [], applied: false };
  }

  // ── Step 0: alias pre-scan (pure string replace, always safe to run) ──
  let prescanned = rawText;
  for (const [pattern, correction] of Object.entries(ALIASES)) {
    const re = new RegExp(escapeRegExp(pattern), "gi");
    if (re.test(prescanned)) {
      corrections[pattern] = correction;
      prescanned = prescanned.replace(re, correction);
    }
  }

  // Without a Groq key we still return the alias-corrected text.
  if (!config.groq.apiKey) {
    return {
      corrected: prescanned,
      corrections,
      medicineTokens: [],
      applied: prescanned !== rawText,
      note: "GROQ_API_KEY not set — applied alias table only (no LLM NER / cleanup).",
    };
  }

  // ── Step 1: Groq medicine NER ──
  const medicineTokens = await extractMedicineTokens(prescanned);

  // ── Step 2: ML normalization of detected tokens ──
  const mlCorrections = await normalizeMedicineTokens(medicineTokens);
  Object.assign(corrections, mlCorrections);

  // ── Step 3: Groq final cleanup with corrections injected ──
  try {
    // Avoid truncating the transcript. Devanagari/Indic scripts use more tokens
    // per character, so scale generously with input length (capped at 8000).
    const cleanupMaxTokens = Math.min(8000, Math.max(2048, Math.ceil(prescanned.length * 2)));
    const cleaned = await groqChat(
      [
        { role: "system", content: buildSystemPrompt(localeId, corrections) },
        { role: "user", content: prescanned },
      ],
      0.1,
      cleanupMaxTokens,
      // Use the stronger model for cleanup: the small model mangles Indic number
      // words and ignores transliteration. Reuses the extraction model.
      config.groq.extractionModel
    );
    if (cleaned) {
      return { corrected: cleaned, corrections, medicineTokens, applied: true };
    }
  } catch (err) {
    console.warn("[correction] Groq cleanup failed:", err instanceof Error ? err.message : err);
  }

  // Cleanup failed — still apply the corrections we gathered as a fallback.
  const fallback = applyCorrections(prescanned, corrections);
  return {
    corrected: fallback,
    corrections,
    medicineTokens,
    applied: fallback !== rawText,
    note: "Groq cleanup unavailable — applied alias + medicine-search corrections directly.",
  };
}

function buildLabeledCleanupPrompt(localeId: string, corrections: Record<string, string>): string {
  const correctionHint =
    Object.keys(corrections).length === 0
      ? ""
      : "\n\nKnown medicine corrections (apply these exactly):\n" +
        Object.entries(corrections)
          .map(([k, v]) => `  "${k}" → "${v}"`)
          .join("\n");

  return `You are a medical transcription corrector specialised for Indian OPD (outpatient department) dictations.
The conversation is spoken ${languageHint(localeId)}.

You are given a transcript ALREADY DIVIDED INTO TURNS by an acoustic speaker-detection
system — one turn per line, each starting with a label like "User 1:" or "User 2:".
These labels come from real audio timestamps, not a guess — they are structurally
correct. Your ONLY job is to correct speech-to-text errors WITHIN each line's text.

Your entire reply must be ONLY the corrected transcript, in the same "User N: text"
line format. No heading, no preamble, no "here is the corrected version", no note
afterward, nothing before the first line or after the last line.

Rules:
1. Preserve the EXACT line structure: same number of lines, same order, same
   "User N:" labels. Never merge, split, reorder, add, or remove a line — even a
   line that looks incomplete (that's a real interruption in the audio). Edit only
   the text after each label's colon.
2. Every word spoken must still appear, in the same line it started in — never
   shorten, summarise, or drop anything.
3. Never add, remove, or infer information that wasn't spoken.
4. Don't translate — a Hindi/regional sentence stays Hindi/regional. The one
   exception is rule 5: fixing the SCRIPT of a word that was already English.
5. Indian speakers code-switch into English mid-sentence constantly, and the STT
   engine often renders that English word phonetically in Devanagari instead of
   Latin script (e.g. "डॉक्टर" is "doctor", "परासिटमोल"/"पैरासिटामोल" is
   "Paracetamol"). Fix every occurrence of this you find, anywhere in the
   transcript — rewrite just that word in Latin script, leave the rest of the
   line untouched. Don't touch words that are genuinely native vocabulary.
6. Fix medicine names specifically (same rule as #5, but call special attention
   to these): Paracetamol, Calpol 650, Dolo 650, Combiflam, Kalcoral D, Augmentin,
   Azithromycin, Amoxicillin, Pantoprazole, Pan-D, Zerodol, Crocin, Allegra,
   Montek LC, Metformin, Amlodipine, Atorvastatin, Ecosprin, Rabeprazole,
   Ondansetron, Domstal, Zolpidem.
7. Normalise vitals/dosage phrasing ("temperature one oh one" → "temperature 101",
   "six fifty mg" → "650 mg") without changing meaning or moving it to another line.
8. If a line is already clean, return it unchanged.${correctionHint}
`;
}

function buildLabeledCleanupWithReferencePrompt(localeId: string, corrections: Record<string, string>): string {
  const correctionHint =
    Object.keys(corrections).length === 0
      ? ""
      : "\n\nKnown medicine corrections (apply these exactly):\n" +
        Object.entries(corrections)
          .map(([k, v]) => `  "${k}" → "${v}"`)
          .join("\n");

  return `You are a medical transcription corrector specialised for Indian OPD (outpatient department) dictations.
The conversation is spoken ${languageHint(localeId)}.

You are given TWO transcripts of the SAME audio, covering the SAME conversation in
the SAME order:

(A) PRIMARY — divided into turns by real acoustic speaker-detection, one turn per
    line, each starting with a label like "User 1:" or "User 2:". These labels come
    from real audio timestamps, not a guess — they are structurally authoritative.
(B) ENGLISH-FORCED REFERENCE — the same audio, transcribed with English forced as
    the decoding language, no line labels. Genuine English speech comes through in
    (B) as real English words. Genuine Hindi/regional speech comes through in (B)
    as rough PHONETIC LATIN-SCRIPT TRANSLITERATION — NOT a translation, so word
    order and content roughly track (A). Use (B) ONLY to tell which words in (A)
    were actually spoken in English (because they appear in (B) as real English at
    the matching point in the conversation) — ignore (B) for everything else.

Your entire reply must be ONLY the corrected version of (A), in the same
"User N: text" line format. No heading, no preamble, no note afterward, nothing
before the first line or after the last line.

Rules:
1. Preserve (A)'s EXACT line structure: same number of lines, same order, same
   "User N:" labels. Never merge, split, reorder, add, or remove a line — even a
   line that looks incomplete (that's a real interruption in the audio). Edit only
   the text after each label's colon.
2. Every word spoken must still appear, in the same line it started in — never
   shorten, summarise, or drop anything.
3. Never add, remove, or infer information that wasn't spoken.
4. Don't translate — a Hindi/regional sentence stays Hindi/regional. The one
   exception is rule 5: fixing the SCRIPT of a word that was already English.
5. Wherever (A) rendered a genuinely-English word/phrase phonetically in Devanagari
   (or another Indic script) — check against (B) — rewrite just that word/phrase in
   its normal English (Latin-script) spelling in (A), leaving the rest of the line
   untouched. Don't touch words that are genuinely native vocabulary, even common
   loanwords Hindi speakers treat as native.
6. Fix medicine names specifically (same rule as #5, but call special attention
   to these): Paracetamol, Calpol 650, Dolo 650, Combiflam, Kalcoral D, Augmentin,
   Azithromycin, Amoxicillin, Pantoprazole, Pan-D, Zerodol, Crocin, Allegra,
   Montek LC, Metformin, Amlodipine, Atorvastatin, Ecosprin, Rabeprazole,
   Ondansetron, Domstal, Zolpidem.
7. Normalise vitals/dosage phrasing ("temperature one oh one" → "temperature 101",
   "six fifty mg" → "650 mg") without changing meaning or moving it to another line.
8. If a line is already clean, return it unchanged.${correctionHint}
`;
}

/**
 * Correct medicine names/STT errors in an already speaker-labeled transcript
 * ("User 1: ...\nUser 2: ..."), preserving the exact line-per-turn structure.
 * Used by acousticDiarize: label first (real timestamps), correct second —
 * correction is a text-level rewrite, so running it after labeling doesn't
 * risk the audio-hallucination problems of transcribing per-turn audio clips.
 *
 * `englishReference`, when given, is a second whole-recording transcript with
 * English forced as the decoding language (see transcribeEnglishForced) — it
 * lets the cleanup pass generalize the Devanagari→Latin script fix to ANY
 * code-switched English word, not just the medicine names in rule 6. Without
 * it, only the explicitly-listed medicine names get fixed reliably; general
 * code-switching fixes are best-effort (the model has no reference to check
 * against, so it under-corrects rather than risk mistranslating).
 */
export async function correctLabeledTranscript(
  labeledText: string,
  localeId: string = config.correction.locale,
  englishReference?: string,
): Promise<CorrectionResult> {
  const corrections: Record<string, string> = {};
  if (!labeledText.trim()) {
    return { corrected: labeledText, corrections, medicineTokens: [], applied: false };
  }

  let prescanned = labeledText;
  for (const [pattern, correction] of Object.entries(ALIASES)) {
    const re = new RegExp(escapeRegExp(pattern), "gi");
    if (re.test(prescanned)) {
      corrections[pattern] = correction;
      prescanned = prescanned.replace(re, correction);
    }
  }
  // Deterministic, not an LLM judgment call — see CODE_SWITCH_ALIASES's comment.
  for (const [pattern, correction] of Object.entries(CODE_SWITCH_ALIASES)) {
    const re = new RegExp(escapeRegExp(pattern), "g");
    if (re.test(prescanned)) {
      corrections[pattern] = correction;
      prescanned = prescanned.replace(re, correction);
    }
  }

  if (!config.groq.apiKey) {
    return {
      corrected: prescanned,
      corrections,
      medicineTokens: [],
      applied: prescanned !== labeledText,
      note: "GROQ_API_KEY not set — applied alias table only (no LLM NER / cleanup).",
    };
  }

  const medicineTokens = await extractMedicineTokens(prescanned);
  Object.assign(corrections, await normalizeMedicineTokens(medicineTokens));

  try {
    const cleanupMaxTokens = Math.min(8000, Math.max(2048, Math.ceil(prescanned.length * 2)));
    const useReference = Boolean(englishReference && englishReference.trim());
    const cleaned = await groqChat(
      [
        {
          role: "system",
          content: useReference
            ? buildLabeledCleanupWithReferencePrompt(localeId, corrections)
            : buildLabeledCleanupPrompt(localeId, corrections),
        },
        {
          role: "user",
          content: useReference
            ? `(A) PRIMARY:\n"""${prescanned}"""\n\n(B) ENGLISH-FORCED REFERENCE:\n"""${englishReference}"""`
            : prescanned,
        },
      ],
      0.1,
      cleanupMaxTokens,
      config.groq.extractionModel
    );
    if (cleaned) {
      return { corrected: cleaned, corrections, medicineTokens, applied: true };
    }
  } catch (err) {
    console.warn("[correction] labeled cleanup failed:", err instanceof Error ? err.message : err);
  }

  const fallback = applyCorrections(prescanned, corrections);
  return {
    corrected: fallback,
    corrections,
    medicineTokens,
    applied: fallback !== labeledText,
    note: "Groq cleanup unavailable — applied alias + medicine-search corrections directly.",
  };
}

function buildCombinePrompt(localeId: string, corrections: Record<string, string>): string {
  const correctionHint =
    Object.keys(corrections).length === 0
      ? ""
      : "\n\nVerified medicine names (use these exact English spellings):\n" +
        Object.entries(corrections)
          .map(([k, v]) => `  "${k}" → "${v}"`)
          .join("\n");

  return `You are an expert MEDICAL TRANSCRIPTIONIST for an Indian OPD (outpatient) clinic.
You are given TWO automatic transcripts of the SAME doctor–patient audio, produced
by two different speech-recognition engines:

(A) REGIONAL — transcribed ${languageHint(localeId)} by an engine strong at the
    local language. TRUST THIS for the CONVERSATION: the natural-language wording,
    sentence flow, symptoms, history, instructions, and anything spoken in the
    local language.

(B) ENGLISH — an English transcript of the same audio by an engine strong at
    English/Latin terms. TRUST THIS for MEDICINE/DRUG NAMES and DOSAGE STRENGTHS
    (e.g. "500 mg") and English medical terminology — which the regional engine
    frequently mis-hears.

Your task: reconcile the two into ONE accurate, clean medical transcript, exactly
as a skilled human scribe who heard the same audio would — using each source for
what it does best and resolving disagreements with clinical judgement.

Rules:
1. Keep the conversation in the regional language/script, following (A). Do NOT
   translate the conversation into English, and do NOT summarise or omit anything
   that was said.
2. For every medicine/drug, write the EXACT drug named in standard English/Latin
   spelling (from (B) and the verified list below) — even where (A) garbled it or
   wrote it in the regional script. Do NOT swap a brand for its generic or vice
   versa (if "Dolo" was said, keep "Dolo").
3. Write ALL medicine dosages and strengths in English digits with units
   (e.g. "500 mg", "250 mg", "40 mg") — NEVER in regional number-words. Normalise
   vitals to digits ("one oh one" / "एक सौ एक" → "101").
4. For frequency, timing and duration that were spoken in the local language,
   TRUST the REGIONAL transcript (A) — e.g. "दो दिन" = "2 days", "दिन में दो बार" =
   "twice a day". The English transcript (B) is only a backup for these.
5. PRESERVE EVERY NUMBER EXACTLY as spoken — never change a count, dose, frequency,
   or duration (e.g. "2 days" must NEVER become "1 day"). State one clear value; do
   NOT produce contradictory or negated phrasing.
6. Fix obvious speech-recognition errors and produce readable, well-punctuated
   text, but stay faithful to the doctor's meaning and wording.
7. NEVER invent or infer clinical facts. Do not add any medicine, diagnosis,
   symptom, vital, or advice that is not present in at least one transcript.
8. Output ONLY the final reconciled transcript — no notes, labels, or commentary.${correctionHint}`;
}

/**
 * Merge a regional-language transcript (accurate conversation) with an English
 * transcript (used to resolve medicine names via FAISS). Medicine detection runs
 * on the ENGLISH text because the FAISS drug DB is Latin-script; the regional
 * text is never used to capture medicines.
 */
export async function combineRegionalAndEnglish(
  regional: string,
  english: string,
  localeId: string = config.correction.locale
): Promise<CorrectionResult> {
  // English selected: the final transcript is the English one (no regional merge).
  if ((localeId || "").slice(0, 2).toLowerCase() === "en") {
    const finalEn = english.trim() || regional.trim();
    return { corrected: finalEn, corrections: {}, medicineTokens: [], applied: true };
  }
  if (!regional.trim()) {
    return { corrected: english.trim(), corrections: {}, medicineTokens: [], applied: false };
  }

  const corrections: Record<string, string> = {};

  // Alias pre-scan on the English transcript.
  let englishScan = english;
  for (const [pattern, correction] of Object.entries(ALIASES)) {
    const re = new RegExp(escapeRegExp(pattern), "gi");
    if (re.test(englishScan)) {
      corrections[pattern] = correction;
      englishScan = englishScan.replace(re, correction);
    }
  }

  let medicineTokens: string[] = [];
  if (config.groq.apiKey) {
    // Step 1+2: medicine NER on the ENGLISH transcript → FAISS canonicalisation.
    medicineTokens = await extractMedicineTokens(englishScan);
    Object.assign(corrections, await normalizeMedicineTokens(medicineTokens));

    // Step 3: merge regional conversation + English medicine names (strong model).
    try {
      const maxTokens = Math.min(8000, Math.max(2048, Math.ceil(regional.length * 2)));
      const merged = await groqChat(
        [
          { role: "system", content: buildCombinePrompt(localeId, corrections) },
          {
            role: "user",
            content:
              `(A) REGIONAL TRANSCRIPT:\n"""${regional}"""\n\n` +
              `(B) ENGLISH TRANSCRIPT:\n"""${englishScan}"""`,
          },
        ],
        0.1,
        maxTokens,
        config.groq.extractionModel
      );
      if (merged) {
        return { corrected: merged, corrections, medicineTokens, applied: true };
      }
    } catch (err) {
      console.warn("[combine] Groq merge failed:", err instanceof Error ? err.message : err);
    }
  }

  // Fallback: regional text with any alias/FAISS corrections applied directly.
  const fallback = applyCorrections(regional, corrections);
  return {
    corrected: fallback,
    corrections,
    medicineTokens,
    applied: fallback !== regional,
    note: config.groq.apiKey
      ? "Groq merge unavailable — applied medicine corrections to the regional transcript."
      : "GROQ_API_KEY not set — returned the regional transcript with alias corrections only.",
  };
}
