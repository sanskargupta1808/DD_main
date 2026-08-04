import {
  emptyExtraction,
  type MedicalExtraction,
  type Prescription,
} from "../types.js";

/**
 * A dependency-free, rule-based medical information extractor.
 *
 * It is intentionally conservative: it favours recall via keyword/pattern
 * matching and never fabricates data. It is NOT a substitute for an LLM or a
 * clinician's review, but it lets VoiceRX produce a useful structured summary
 * with zero configuration.
 */

const SYMPTOM_KEYWORDS = [
  "fever", "cough", "cold", "headache", "migraine", "nausea", "vomiting",
  "diarrhea", "diarrhoea", "constipation", "fatigue", "tiredness", "weakness",
  "dizziness", "chest pain", "shortness of breath", "breathlessness",
  "sore throat", "runny nose", "congestion", "body ache", "back pain",
  "joint pain", "muscle pain", "abdominal pain", "stomach pain", "stomach ache",
  "chills", "sweating", "rash", "itching", "swelling", "bleeding",
  "blurred vision", "palpitations", "insomnia", "loss of appetite",
  "weight loss", "weight gain", "anxiety", "depression", "cramps", "burning",
  "numbness", "tingling", "wheezing", "sneezing", "sneezing",
];

const DISEASE_KEYWORDS = [
  "diabetes", "hypertension", "high blood pressure", "low blood pressure",
  "asthma", "pneumonia", "bronchitis", "covid", "covid-19", "influenza", "flu",
  "migraine", "anemia", "anaemia", "arthritis", "gastritis", "ulcer",
  "infection", "uti", "urinary tract infection", "tuberculosis", "tb",
  "hypothyroidism", "hyperthyroidism", "thyroid", "cholesterol",
  "heart disease", "cardiac", "stroke", "cancer", "tumor", "tumour",
  "allergy", "sinusitis", "tonsillitis", "dengue", "malaria", "typhoid",
  "jaundice", "hepatitis", "kidney stone", "depression", "anxiety disorder",
  "pcos", "pcod", "vitamin d deficiency", "vitamin b12 deficiency",
  "viral fever", "throat infection", "ear infection", "conjunctivitis",
];

const DIAGNOSIS_TRIGGERS = [
  "diagnosed with", "diagnosis is", "diagnosis of", "suffering from",
  "you have", "this is", "looks like", "appears to be", "consistent with",
  "it could be", "seems to be", "we think it's", "indicative of",
];

// Single regex that finds every prescription cue anywhere in the transcript.
// Live transcripts are usually one long run-on line, so we cannot rely on
// sentence boundaries — we scan globally and treat the span up to the next cue
// as one prescription.
const PRESCRIPTION_TRIGGER_RE =
  /\b(?:prescrib(?:e|ed|ing)|i'?ll give you|i will give you|i'?m giving you|giving you|give you|given you|put you on|i want you to take|starting|start|taking|take|continue|recommend|apply)\b/gi;

// Words that follow a cue but are never medications (generic advice etc.).
const NON_MEDICATION = new Set([
  "rest", "fluids", "fluid", "water", "care", "plenty", "lots", "lot",
  "it", "them", "these", "those", "this", "that", "your", "you", "medicine",
  "medicines", "medication", "tablet", "tablets", "the", "a", "an", "some",
  "and", "also", "as", "to", "of", "if", "when", "after", "before",
]);

// Leading filler words to skip before the medication name.
const LEADING_SKIP = new Set([
  "you", "your", "the", "a", "an", "some", "him", "her", "them", "with",
  "on", "taking", "tablet", "tablets", "capsule", "capsules", "of",
]);

const DOSAGE_RE =
  /\b(\d+(?:\.\d+)?)\s?-?\s?(mg|milligrams?|mcg|micrograms?|ml|millilitres?|milliliters?|g|grams?|tablets?|tabs?|capsules?|caps?|pills?|puffs?|drops?|units?|sprays?|teaspoons?|tsp|tbsp)\b/gi;

const FREQUENCY_RE = new RegExp(
  [
    "once (?:a day|daily)",
    "twice (?:a day|daily)",
    "thrice (?:a day|daily)",
    "three times (?:a day|daily)",
    "four times (?:a day|daily)",
    "every \\d+ ?(?:hours?|hrs?|hr)",
    "\\bq\\d+h\\b",
    "\\b(?:od|bd|bid|tds|tid|qid|qhs|hs|sos|prn)\\b",
    "(?:in the )?morning",
    "(?:in the )?afternoon",
    "(?:in the )?evening",
    "(?:at )?night",
    "at bedtime",
    "before (?:meals?|food|breakfast|lunch|dinner)",
    "after (?:meals?|food|breakfast|lunch|dinner)",
    "with (?:meals?|food)",
    "empty stomach",
    "as needed",
    "when (?:needed|required)",
    "every (?:morning|night|day)",
    "weekly",
    "alternate days?",
  ].join("|"),
  "gi"
);

const DURATION_RE =
  /\bfor\s+(?:the\s+next\s+)?(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fourteen|fifteen|twenty|thirty|couple of|few)\s+(day|days|week|weeks|month|months|night|nights)\b/gi;

// Numbers may be spoken as words ("five days") since live transcripts rarely
// contain digits/punctuation.
const NUM = "\\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fourteen|fifteen|twenty|thirty|couple of|few";

const NEXT_VISIT_RE = new RegExp(
  "\\b(?:come back|follow[\\s-]?up|next visit|next appointment|see (?:you|me)|review|revisit|check ?up again|return|visit again)\\b" +
    "[^.?!]{0,40}?" +
    "\\b((?:in|after|within)\\s+(?:" +
    NUM +
    ")\\s+(?:days?|weeks?|months?)" +
    "|next\\s+(?:week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)" +
    "|tomorrow" +
    "|on\\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)" +
    "|on\\s+\\d{1,2}(?:st|nd|rd|th)?(?:\\s+\\w+)?)",
  "i"
);

const ALLERGY_RE = /\ballergic to\s+([a-z0-9 ,\-]+?)(?:[.?!]|,|\band\b|$)/gi;

const VITALS_RES: RegExp[] = [
  /\bblood pressure (?:is |of |was )?\d{2,3}\s?\/\s?\d{2,3}\b/gi,
  /\bbp (?:is |of |was )?\d{2,3}\s?\/\s?\d{2,3}\b/gi,
  /\b\d{2,3}\s?\/\s?\d{2,3}\s?(?:mmhg)\b/gi,
  /\btemperature (?:is |of |was )?\d{2,3}(?:\.\d)?\s?(?:°|degrees?)?\s?(?:f|c|fahrenheit|celsius)?\b/gi,
  /\b(?:pulse|heart rate)(?: is| of| was)?\s?\d{2,3}\s?(?:bpm)?\b/gi,
  /\b(?:blood )?sugar(?: level)?(?: is| of| was)?\s?\d{2,3}\b/gi,
  /\bspo2(?: is| of| was)?\s?\d{2,3}\s?%?\b/gi,
  /\boxygen (?:saturation|level)(?: is| of| was)?\s?\d{2,3}\s?%?\b/gi,
];

function splitSentences(text: string): string[] {
  return text
    .replace(/\s+/g, " ")
    .split(/(?<=[.?!])\s+|\n+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function uniq(arr: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const item of arr) {
    const key = item.toLowerCase().trim();
    if (key && !seen.has(key)) {
      seen.add(key);
      out.push(item.trim());
    }
  }
  return out;
}

function findKeywords(text: string, keywords: string[]): string[] {
  const lower = text.toLowerCase();
  const found: string[] = [];
  for (const kw of keywords) {
    const re = new RegExp(`\\b${kw.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i");
    if (re.test(lower)) found.push(kw);
  }
  return found;
}

function titleCase(s: string): string {
  return s.replace(/\b\w/g, (c) => c.toUpperCase());
}

function extractPatient(text: string): MedicalExtraction["patient"] {
  const patient: MedicalExtraction["patient"] = {};

  const age = text.match(/\b(\d{1,3})[\s-]?(?:years?|yrs?|year)[\s-]?(?:old|of age)?\b/i);
  if (age) patient.age = `${age[1]} years`;

  if (/\b(she|her|woman|female|lady|mrs\.?|ms\.?|girl)\b/i.test(text)) patient.gender = "Female";
  else if (/\b(he|his|him|man|male|gentleman|mr\.?|boy)\b/i.test(text)) patient.gender = "Male";

  const name = text.match(/\b(?:patient(?:'s)? name is|this is|mr\.?|mrs\.?|ms\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)/);
  if (name) patient.name = name[1];

  return patient;
}

function extractPrescriptions(transcript: string): Prescription[] {
  const out: Prescription[] = [];
  const triggers = [...transcript.matchAll(PRESCRIPTION_TRIGGER_RE)];

  for (let i = 0; i < triggers.length; i++) {
    const m = triggers[i];
    const start = (m.index ?? 0) + m[0].length;
    const nextStart = i + 1 < triggers.length ? triggers[i + 1].index ?? transcript.length : transcript.length;
    // The medication + its details live between this cue and the next one.
    const segment = transcript.slice(start, Math.min(nextStart, start + 160)).replace(/^[\s,:-]+/, "");

    // Build the medication name: skip leading filler, then take up to 3 words,
    // stopping at the first number or stop-word.
    const tokens = segment.split(/\s+/);
    let idx = 0;
    while (idx < tokens.length && LEADING_SKIP.has(tokens[idx].toLowerCase().replace(/[^a-z]/g, ""))) idx++;

    const medWords: string[] = [];
    for (; idx < tokens.length && medWords.length < 3; idx++) {
      const clean = tokens[idx].toLowerCase().replace(/[^a-z0-9'+\-]/g, "");
      if (!clean) break;
      if (/\d/.test(clean)) break; // hit a dosage number
      if (NON_MEDICATION.has(clean)) break;
      medWords.push(tokens[idx].replace(/[^A-Za-z0-9'+\-]/g, ""));
    }
    let medication = medWords.join(" ").trim();
    if (!medication || medication.length < 3) continue;

    const dosage = (segment.match(DOSAGE_RE) ?? [])[0];
    const frequency = (segment.match(FREQUENCY_RE) ?? [])[0];
    const duration = (segment.match(DURATION_RE) ?? [])[0];

    // Require at least one structured detail so generic advice
    // ("take rest", "continue your diet") is not logged as a prescription.
    if (!dosage && !frequency && !duration) continue;

    const instrParts: string[] = [];
    const instr = segment.match(/\b(?:after|before|with)\s+(?:meals?|food|breakfast|lunch|dinner)\b/i);
    if (instr) instrParts.push(instr[0]);
    if (/empty stomach/i.test(segment)) instrParts.push("empty stomach");

    out.push({
      medication: titleCase(medication),
      dosage: dosage?.trim(),
      frequency: frequency?.trim(),
      duration: duration?.trim(),
      instructions: instrParts.length ? instrParts.join(", ") : undefined,
    });
  }

  // De-dupe by medication name (keep the richest entry).
  const byMed = new Map<string, Prescription>();
  for (const p of out) {
    const key = p.medication.toLowerCase();
    const existing = byMed.get(key);
    if (!existing) {
      byMed.set(key, p);
    } else {
      byMed.set(key, {
        medication: p.medication,
        dosage: existing.dosage ?? p.dosage,
        frequency: existing.frequency ?? p.frequency,
        duration: existing.duration ?? p.duration,
        instructions: existing.instructions ?? p.instructions,
      });
    }
  }
  return [...byMed.values()];
}

export function heuristicExtract(transcript: string): MedicalExtraction {
  const result = emptyExtraction();
  if (!transcript.trim()) return result;

  const sentences = splitSentences(transcript);

  result.patient = extractPatient(transcript);

  // Symptoms
  result.symptoms = uniq(findKeywords(transcript, SYMPTOM_KEYWORDS).map(titleCase));

  // Diagnoses: keyword diseases + phrases following diagnosis triggers.
  const diagnoses: string[] = findKeywords(transcript, DISEASE_KEYWORDS).map(titleCase);
  const DIAG_STOP = new Set([
    "i", "i'll", "i'm", "i've", "we", "we'll", "you", "so", "and", "but",
    "take", "taking", "prescribe", "prescribing", "give", "giving", "come",
    "let", "let's", "now", "then", "also", "start", "starting", "for",
  ]);
  for (const sentence of sentences) {
    const lower = sentence.toLowerCase();
    for (const trig of DIAGNOSIS_TRIGGERS) {
      const at = lower.indexOf(trig);
      if (at === -1) continue;
      const rawPhrase = sentence
        .slice(at + trig.length)
        .replace(/^[\s,:-]+/, "")
        .split(/[.?!,]/)[0]
        .replace(/^(a|an|the)\s+/i, "")
        .trim();
      // In run-on transcripts there's no punctuation, so cut at the first
      // sentence-connective / next-instruction word and cap the length.
      const words: string[] = [];
      for (const w of rawPhrase.split(/\s+/)) {
        if (DIAG_STOP.has(w.toLowerCase())) break;
        words.push(w);
        if (words.length >= 5) break;
      }
      const phrase = words.join(" ").trim();
      if (phrase && phrase.length <= 40 && /[a-z]/i.test(phrase)) {
        diagnoses.push(phrase);
      }
    }
  }
  result.diagnoses = uniq(diagnoses);

  // Prescriptions
  result.prescriptions = extractPrescriptions(transcript);

  // Follow-up
  const fu = transcript.match(NEXT_VISIT_RE);
  if (fu) {
    result.followUp.nextVisit = fu[1].trim();
    result.followUp.instructions = fu[0].trim();
  }

  // Allergies
  const allergies: string[] = [];
  for (const m of transcript.matchAll(ALLERGY_RE)) {
    if (m[1]) allergies.push(titleCase(m[1].trim()));
  }
  result.allergies = uniq(allergies);

  // Vitals
  const vitals: string[] = [];
  for (const re of VITALS_RES) {
    for (const m of transcript.matchAll(re)) vitals.push(m[0].trim());
  }
  result.vitals = uniq(vitals);

  return result;
}
