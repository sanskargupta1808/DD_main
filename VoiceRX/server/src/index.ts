import cors from "cors";
import express from "express";
import { config } from "./config.js";
import { correctRouter } from "./routes/correct.js";
import { extractRouter } from "./routes/extract.js";
import { transcribeRouter } from "./routes/transcribe.js";
import { combineRouter } from "./routes/combine.js";
import { diarizeRouter } from "./routes/diarize.js";
import { medicineServiceHealthy } from "./services/medicineSearch.js";

const app = express();

app.use(
  cors({
    origin: config.clientOrigins,
  })
);
// Transcripts can be long; allow a generous JSON body.
app.use(express.json({ limit: "2mb" }));

app.get("/", (_req, res) => {
  res.type("text").send(
    "VoiceRX API is running. Use /api/health, /api/correct, or /api/extract."
  );
});

app.get("/api/health", async (_req, res) => {
  res.json({
    status: "ok",
    extractionProvider: config.extractionProvider,
    transcription: {
      provider: config.transcription.provider,
      dual: config.transcription.dual,
      model:
        config.transcription.provider === "bhashini"
          ? "ai4bharat/IndicConformer (via Bhashini)"
          : config.transcription.provider === "custom"
            ? config.transcription.model || "whisper-1"
            : config.groq.transcriptionModel,
      customBaseUrl: config.transcription.provider === "custom" ? config.transcription.baseUrl : undefined,
      bhashiniConfigured:
        config.transcription.provider === "bhashini"
          ? Boolean(config.bhashini.userId && config.bhashini.ulcaApiKey)
          : undefined,
    },
    correction: {
      enabled: config.correction.enabled,
      locale: config.correction.locale,
      groqConfigured: Boolean(config.groq.apiKey),
      groqModel: config.groq.model,
      medicineServiceUrl: config.medicineSearch.url,
      medicineServiceReachable: await medicineServiceHealthy(),
    },
  });
});

app.use("/api/extract", extractRouter);
app.use("/api/correct", correctRouter);
app.use("/api/transcribe", transcribeRouter);
app.use("/api/combine", combineRouter);
app.use("/api/diarize", diarizeRouter);

// Fallback 404 for unknown API routes.
app.use("/api", (_req, res) => res.status(404).json({ error: "Not found" }));

app.listen(config.port, config.host, () => {
  console.log(`VoiceRX server listening on http://${config.host}:${config.port}`);
  console.log(`  extraction provider: ${config.extractionProvider}`);
  console.log(
    `  correction: ${config.correction.enabled ? "on" : "off"}` +
      ` (groq ${config.groq.apiKey ? "configured" : "NOT configured"}, ` +
      `medicine service: ${config.medicineSearch.url})`
  );
  console.log(
    "  NOTE: API is unauthenticated — add auth before exposing this beyond localhost."
  );
});
