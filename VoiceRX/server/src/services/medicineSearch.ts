import { config } from "../config.js";

export interface MedicineMeta {
  id: number;
  brand_name: string;
  generic_name: string;
  strength: string;
  form: string;
}

export interface MedicineSearchResult {
  query: string;
  normalized_query: string;
  match: string | null;
  score: number;
  confidence: "high" | "low" | "bktree" | "none";
  medicine: MedicineMeta | null;
  elapsed_ms: number;
}

/**
 * Client for the FAISS medicine-search FastAPI service
 * (medicine_pipeline/server.py). Ports the Dart `MedicineSearchService`.
 *
 * Returns null on any failure (service down, timeout, bad response) so the
 * caller can degrade gracefully.
 */
export async function searchMedicine(query: string): Promise<MedicineSearchResult | null> {
  const q = query.trim();
  if (!q) return null;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), config.medicineSearch.timeoutMs);
  try {
    const res = await fetch(`${config.medicineSearch.url}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q }),
      signal: controller.signal,
    });
    if (!res.ok) return null;
    return (await res.json()) as MedicineSearchResult;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

export function isHighConfidence(result: MedicineSearchResult): boolean {
  return result.confidence === "high" || result.score >= config.medicineSearch.confidenceThreshold;
}

/** Quick reachability probe for the medicine-search service. */
export async function medicineServiceHealthy(): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), config.medicineSearch.timeoutMs);
  try {
    const res = await fetch(`${config.medicineSearch.url}/health`, { signal: controller.signal });
    if (!res.ok) return false;
    const data: any = await res.json();
    return data?.status === "ok";
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}
