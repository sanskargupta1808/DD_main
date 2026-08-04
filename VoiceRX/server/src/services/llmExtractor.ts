import {
  BedrockRuntimeClient,
  InvokeModelCommand,
} from "@aws-sdk/client-bedrock-runtime";
import { config } from "../config.js";
import { emptyExtraction, type MedicalExtraction } from "../types.js";

const SYSTEM_PROMPT = `You are a clinical scribe assistant. You read a transcript of a
doctor-patient conversation and extract structured information. You ONLY extract
information that is explicitly present in the transcript — never invent or infer
clinical facts. You are a documentation aid, not a diagnostic tool.

Return ONLY a JSON object with exactly this shape (no markdown, no commentary):
{
  "patient": { "name": string|null, "age": string|null, "gender": string|null },
  "symptoms": string[],
  "diagnoses": string[],
  "prescriptions": [
    {
      "medication": string,
      "dosage": string|null,      // amount per dose, e.g. "500 mg"
      "frequency": string|null,   // timing, e.g. "twice daily after meals"
      "duration": string|null,    // e.g. "for 5 days"
      "instructions": string|null
    }
  ],
  "followUp": { "nextVisit": string|null, "instructions": string|null },
  "allergies": string[],
  "vitals": string[],
  "notes": string[]
}
Use empty arrays / null where information is absent.

LANGUAGE — very important:
- Output EVERY field value in ENGLISH. Translate all clinical details from the
  transcript's language into clear clinical English — symptoms ("खांसी" → "Cough",
  "जुकाम" → "Cold"), diagnoses, frequency ("दिन में दो बार" → "twice a day",
  "भोजन के बाद" → "after meals"), duration ("तीन दिन के लिए" → "for 3 days"),
  instructions, follow-up, vitals and notes.
- Transliterate the patient's name to Latin script ("संस्कार" → "Sanskar").

MEDICATION — very important:
- Use the EXACT drug the doctor named, written in standard English/Latin spelling
  ("परासिटामोल" → "Paracetamol", "डोलो" → "Dolo").
- Do NOT replace a brand with its generic or vice versa. If the doctor said "Dolo",
  output "Dolo" (NOT "Paracetamol"). Keep each medicine exactly as spoken.

NUMBERS — very important:
- Preserve every number EXACTLY as in the transcript. Translate number-words to the
  same digit ("दो" → "2", "तीन" → "3", "पाँच" → "5"). NEVER change a quantity, dose,
  frequency count, or duration (e.g. "दो दिन" → "for 2 days", never "1 day").`;

/** Coerce arbitrary parsed JSON into a well-formed MedicalExtraction. */
function coerce(raw: unknown): MedicalExtraction {
  const out = emptyExtraction();
  if (!raw || typeof raw !== "object") return out;
  const o = raw as Record<string, any>;

  const strArr = (v: any): string[] =>
    Array.isArray(v) ? v.filter((x) => typeof x === "string" && x.trim()).map((x) => x.trim()) : [];
  const str = (v: any): string | undefined =>
    typeof v === "string" && v.trim() ? v.trim() : undefined;

  if (o.patient && typeof o.patient === "object") {
    out.patient = {
      name: str(o.patient.name),
      age: str(o.patient.age),
      gender: str(o.patient.gender),
    };
  }
  out.symptoms = strArr(o.symptoms);
  out.diagnoses = strArr(o.diagnoses);
  out.allergies = strArr(o.allergies);
  out.vitals = strArr(o.vitals);
  out.notes = strArr(o.notes);

  if (Array.isArray(o.prescriptions)) {
    out.prescriptions = o.prescriptions
      .filter((p: any) => p && typeof p === "object" && str(p.medication))
      .map((p: any) => ({
        medication: str(p.medication)!,
        dosage: str(p.dosage),
        frequency: str(p.frequency),
        duration: str(p.duration),
        instructions: str(p.instructions),
      }));
  }

  if (o.followUp && typeof o.followUp === "object") {
    out.followUp = {
      nextVisit: str(o.followUp.nextVisit),
      instructions: str(o.followUp.instructions),
    };
  }
  return out;
}

/** Pull the first balanced JSON object out of a model response. */
function parseJsonObject(text: string): unknown {
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start === -1 || end === -1 || end < start) {
    throw new Error("No JSON object found in model response");
  }
  return JSON.parse(text.slice(start, end + 1));
}

interface OpenAICompatibleOptions {
  apiKey: string;
  baseUrl: string;
  model: string;
  /** Provider label used in error messages. */
  label: string;
}

/**
 * Shared extractor for any OpenAI-compatible chat-completions API
 * (OpenAI, Groq, Together, local servers, …). Requests JSON output and coerces
 * it into a MedicalExtraction.
 */
async function openAICompatibleExtract(
  transcript: string,
  opts: OpenAICompatibleOptions
): Promise<MedicalExtraction> {
  if (!opts.apiKey) {
    throw new Error(`${opts.label} API key is not set`);
  }
  const res = await fetch(`${opts.baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${opts.apiKey}`,
    },
    body: JSON.stringify({
      model: opts.model,
      temperature: 0,
      max_tokens: 4096,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: `Transcript:\n"""${transcript}"""` },
      ],
    }),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${opts.label} API error ${res.status}: ${body.slice(0, 300)}`);
  }
  const data: any = await res.json();
  const content = data?.choices?.[0]?.message?.content ?? "";
  return coerce(parseJsonObject(content));
}

export async function openaiExtract(transcript: string): Promise<MedicalExtraction> {
  return openAICompatibleExtract(transcript, {
    apiKey: config.openai.apiKey,
    baseUrl: config.openai.baseUrl,
    model: config.openai.model,
    label: "OpenAI",
  });
}

export async function groqExtract(transcript: string): Promise<MedicalExtraction> {
  return openAICompatibleExtract(transcript, {
    apiKey: config.groq.apiKey,
    baseUrl: config.groq.baseUrl,
    model: config.groq.extractionModel,
    label: "Groq",
  });
}

export async function bedrockExtract(transcript: string): Promise<MedicalExtraction> {
  const client = new BedrockRuntimeClient({ region: config.bedrock.region });
  const body = {
    anthropic_version: "bedrock-2023-05-31",
    max_tokens: 2048,
    temperature: 0,
    system: SYSTEM_PROMPT,
    messages: [
      {
        role: "user",
        content: [{ type: "text", text: `Transcript:\n"""${transcript}"""` }],
      },
    ],
  };

  const command = new InvokeModelCommand({
    modelId: config.bedrock.modelId,
    contentType: "application/json",
    accept: "application/json",
    body: JSON.stringify(body),
  });

  const response = await client.send(command);
  const decoded = JSON.parse(new TextDecoder().decode(response.body));
  const text: string = decoded?.content?.[0]?.text ?? "";
  return coerce(parseJsonObject(text));
}
