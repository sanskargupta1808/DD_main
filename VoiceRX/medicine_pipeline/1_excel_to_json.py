#!/usr/bin/env python3
"""
Step 1: Convert Medicine.xlsx → medicines.json
Parses brand name, generic names, strengths, dosage form, and builds search_text.
Run from the repo root:
    python3 tools/medicine_pipeline/1_excel_to_json.py
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
EXCEL_PATH   = Path("lib/Medicine.xlsx")
OUT_JSON     = Path("tools/medicine_pipeline/output/medicines.json")
OUT_META     = Path("tools/medicine_pipeline/output/medicine_meta.json")

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

# ── Dosage form keywords (checked against brand name suffix) ──────────────────
FORM_KEYWORDS = [
    "Tablet", "Tablets", "Tab", "Capsule", "Capsules", "Cap",
    "Syrup", "Suspension", "Solution", "Drops", "Drop",
    "Injection", "Inj", "Infusion", "Gel", "Cream", "Ointment",
    "Lotion", "Spray", "Inhaler", "Inhaler", "Patch", "Sachet",
    "Powder", "Granules", "Suppository", "Eye Drop", "Ear Drop",
    "Nasal Spray", "Mouth Wash", "Linctus", "Paste", "Foam",
]

# Simple therapeutic category map: generic keyword → category hint
THERAPEUTIC_HINTS: dict[str, str] = {
    "paracetamol":       "Fever Pain Analgesic Antipyretic",
    "ibuprofen":         "Pain Fever Anti-inflammatory NSAID",
    "aceclofenac":       "Pain Anti-inflammatory NSAID",
    "diclofenac":        "Pain Anti-inflammatory NSAID",
    "amoxicillin":       "Antibiotic Infection",
    "azithromycin":      "Antibiotic Infection",
    "cefixime":          "Antibiotic Infection",
    "ceftriaxone":       "Antibiotic Infection",
    "ciprofloxacin":     "Antibiotic Infection",
    "metronidazole":     "Antibiotic Infection Anaerobic",
    "doxycycline":       "Antibiotic Infection",
    "clindamycin":       "Antibiotic Infection",
    "pantoprazole":      "Acidity GERD PPI Gastric",
    "omeprazole":        "Acidity GERD PPI Gastric",
    "rabeprazole":       "Acidity GERD PPI Gastric",
    "ondansetron":       "Vomiting Nausea Antiemetic",
    "domperidone":       "Vomiting Nausea Antiemetic Gastric",
    "metformin":         "Diabetes Antidiabetic Blood Sugar",
    "glimepiride":       "Diabetes Antidiabetic Blood Sugar",
    "insulin":           "Diabetes Antidiabetic Blood Sugar",
    "amlodipine":        "Hypertension Blood Pressure Calcium Channel Blocker",
    "atorvastatin":      "Cholesterol Statin Lipid",
    "rosuvastatin":      "Cholesterol Statin Lipid",
    "losartan":          "Hypertension Blood Pressure ARB",
    "telmisartan":       "Hypertension Blood Pressure ARB",
    "salbutamol":        "Asthma Bronchodilator Breathing",
    "montelukast":       "Asthma Allergy Anti-allergic",
    "cetirizine":        "Allergy Antihistamine",
    "fexofenadine":      "Allergy Antihistamine",
    "levothyroxine":     "Thyroid Hypothyroid",
    "calcium":           "Calcium Bone Supplement",
    "vitamin d":         "Vitamin Supplement Bone",
    "iron":              "Anaemia Iron Supplement",
    "folic acid":        "Anaemia Supplement Pregnancy",
    "tramadol":          "Pain Opioid Analgesic",
    "pregabalin":        "Nerve Pain Neuropathy",
    "gabapentin":        "Nerve Pain Neuropathy",
    "methylprednisolone":"Steroid Anti-inflammatory",
    "dexamethasone":     "Steroid Anti-inflammatory",
    "prednisolone":      "Steroid Anti-inflammatory",
    "hydroxychloroquine":"Arthritis Anti-rheumatic",
    "levocetirizine":    "Allergy Antihistamine",
    "ranitidine":        "Acidity H2 Blocker Gastric",
    "clopidogrel":       "Heart Blood Thinner Antiplatelet",
    "aspirin":           "Heart Blood Thinner Antiplatelet Fever Pain",
    "atenolol":          "Heart Beta Blocker Hypertension",
    "metoprolol":        "Heart Beta Blocker Hypertension",
    "furosemide":        "Diuretic Heart Kidney Fluid",
    "spironolactone":    "Diuretic Heart Kidney Fluid",
    "sildenafil":        "Erectile Dysfunction",
    "tadalafil":         "Erectile Dysfunction",
    "alprazolam":        "Anxiety Benzodiazepine Sleep",
    "clonazepam":        "Epilepsy Anxiety Benzodiazepine",
    "escitalopram":      "Depression Anxiety SSRI",
    "sertraline":        "Depression Anxiety SSRI",
    "amitriptyline":     "Depression Neuropathy",
}


def extract_form(brand_name: str) -> str:
    """Detect dosage form from the brand name string."""
    bn = brand_name.lower()
    for form in FORM_KEYWORDS:
        if form.lower() in bn:
            return form.replace(" ", " ").title()
    return ""


def parse_composition(comp: str) -> tuple[list[str], list[str]]:
    """
    Given  'Domperidone (10mg) + Paracetamol (500mg) + Tramadol (50mg)'
    Returns:
        generics  = ['Domperidone', 'Paracetamol', 'Tramadol']
        strengths = ['10mg', '500mg', '50mg']
    """
    generics: list[str] = []
    strengths: list[str] = []
    # Split on ' + ' or ' +'
    parts = re.split(r'\s*\+\s*', comp)
    for part in parts:
        part = part.strip()
        m = re.match(r'^(.+?)\s*\(([^)]+)\)\s*$', part)
        if m:
            generics.append(m.group(1).strip())
            strengths.append(m.group(2).strip())
        else:
            # No parenthesised strength
            generics.append(part)
    return generics, strengths


def build_therapeutic_hint(generics: list[str]) -> str:
    hints: list[str] = []
    for g in generics:
        g_lower = g.lower()
        for key, hint in THERAPEUTIC_HINTS.items():
            if key in g_lower:
                hints.append(hint)
                break
    return " ".join(dict.fromkeys(hints))  # deduplicated, order-preserved


def build_search_text(brand: str, generics: list[str], strengths: list[str],
                      form: str, therapeutic: str) -> str:
    parts = [brand]
    parts.extend(generics)
    parts.extend(strengths)
    if form:
        parts.append(form)
    if therapeutic:
        parts.append(therapeutic)
    return " ".join(filter(None, parts))


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"[1/4] Reading {EXCEL_PATH} …")
    df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]
    df = df.fillna("")
    print(f"      {len(df):,} rows loaded.")

    records: list[dict] = []
    meta: list[dict] = []

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        brand: str   = str(row.Medicine).strip()
        comp: str    = str(row.Composition).strip()
        mfr: str     = str(row.ManufacturerName).strip()

        generics, strengths = parse_composition(comp)
        form        = extract_form(brand)
        therapeutic = build_therapeutic_hint(generics)
        search_text = build_search_text(brand, generics, strengths, form, therapeutic)

        rec = {
            "id":           idx,
            "brand_name":   brand,
            "generic_name": " + ".join(generics),
            "strength":     " + ".join(strengths),
            "form":         form,
            "manufacturer": mfr,
            "composition":  comp,
            "search_text":  search_text,
        }
        records.append(rec)

        # Lightweight meta (id + names only, for BK-Tree / lookup)
        meta.append({
            "id":         idx,
            "brand_name": brand,
            "generic_name": " + ".join(generics),
            "strength":   " + ".join(strengths),
            "form":       form,
        })

        if idx % 50_000 == 0:
            print(f"      … processed {idx:,} / {len(df):,}")

    print(f"[2/4] Writing {OUT_JSON} …")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[3/4] Writing {OUT_META} …")
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[4/4] Done! {len(records):,} medicines converted.")
    print(f"      Sample entry:")
    import pprint
    pprint.pprint(records[100])


if __name__ == "__main__":
    main()
