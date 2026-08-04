// Shared domain types for VoiceRX.
// NOTE: this file is duplicated verbatim in client/src/types.ts.
// If you change the schema, update both copies.

export interface PatientInfo {
  name?: string;
  age?: string;
  gender?: string;
}

export interface Prescription {
  /** Drug / medication name */
  medication: string;
  /** Amount per dose, e.g. "500 mg", "2 tablets" */
  dosage?: string;
  /** How often / when to take it, e.g. "twice daily", "after meals" */
  frequency?: string;
  /** How long to take it, e.g. "for 5 days" */
  duration?: string;
  /** Any extra instructions */
  instructions?: string;
}

export interface FollowUp {
  /** Next visit date — absolute or relative as spoken, e.g. "in 2 weeks" */
  nextVisit?: string;
  instructions?: string;
}

export interface MedicalExtraction {
  patient: PatientInfo;
  symptoms: string[];
  diagnoses: string[];
  prescriptions: Prescription[];
  followUp: FollowUp;
  allergies: string[];
  vitals: string[];
  notes: string[];
}

export interface ExtractRequest {
  transcript: string;
  /** Override whether to run STT correction first (defaults to server config). */
  correct?: boolean;
  /** STT locale for the correction prompt, e.g. en_IN, hi_IN, gu_IN. */
  locale?: string;
}

export interface ExtractResponse {
  extraction: MedicalExtraction;
  /** Which engine produced this result: heuristic | openai | bedrock */
  provider: string;
  /** Present when the result is degraded (e.g. LLM failed, fell back to heuristic) */
  warning?: string;
  /** The transcript after STT correction (present when correction ran). */
  correctedTranscript?: string;
  /** Map of original phrase/token → corrected name applied during correction. */
  corrections?: Record<string, string>;
  /** Note about correction status (e.g. skipped because no Groq key). */
  correctionNote?: string;
}

export function emptyExtraction(): MedicalExtraction {
  return {
    patient: {},
    symptoms: [],
    diagnoses: [],
    prescriptions: [],
    followUp: {},
    allergies: [],
    vitals: [],
    notes: [],
  };
}
