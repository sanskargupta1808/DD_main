// Languages offered in the UI. `locale` is sent to the server, which derives
// the Whisper language code (first 2 chars) and the correction-prompt locale.
// `speechLang` is the BCP-47 tag for the optional browser live-preview.
export interface LanguageOption {
  locale: string; // "auto" | "en_IN" | "hi_IN" | …
  label: string;
  speechLang: string;
}

export const LANGUAGES: LanguageOption[] = [
  { locale: "auto", label: "Auto-detect", speechLang: "en-IN" },
  { locale: "en_IN", label: "English (India)", speechLang: "en-IN" },
  { locale: "hi_IN", label: "हिन्दी · Hindi", speechLang: "hi-IN" },
  { locale: "gu_IN", label: "ગુજરાતી · Gujarati", speechLang: "gu-IN" },
  { locale: "mr_IN", label: "मराठी · Marathi", speechLang: "mr-IN" },
  { locale: "bn_IN", label: "বাংলা · Bengali", speechLang: "bn-IN" },
  { locale: "ta_IN", label: "தமிழ் · Tamil", speechLang: "ta-IN" },
  { locale: "te_IN", label: "తెలుగు · Telugu", speechLang: "te-IN" },
  { locale: "kn_IN", label: "ಕನ್ನಡ · Kannada", speechLang: "kn-IN" },
  { locale: "ml_IN", label: "മലയാളം · Malayalam", speechLang: "ml-IN" },
  { locale: "or_IN", label: "ଓଡ଼ିଆ · Odia", speechLang: "or-IN" },
  { locale: "pa_IN", label: "ਪੰਜਾਬੀ · Punjabi", speechLang: "pa-IN" },
];

export function speechLangFor(locale: string): string {
  return LANGUAGES.find((l) => l.locale === locale)?.speechLang ?? "en-IN";
}
