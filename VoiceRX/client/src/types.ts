// Shared domain types for VoiceRX.
// NOTE: this file is duplicated verbatim from server/src/types.ts.
// If you change the schema, update both copies.

export interface PatientInfo {
  name?: string;
  age?: string;
  gender?: string;
}

export interface Prescription {
  medication: string;
  dosage?: string;
  frequency?: string;
  duration?: string;
  instructions?: string;
}

export interface FollowUp {
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

export interface ExtractResponse {
  extraction: MedicalExtraction;
  provider: string;
  warning?: string;
  correctedTranscript?: string;
  corrections?: Record<string, string>;
  correctionNote?: string;
}
