import { useEffect, useRef, useState, type ChangeEvent } from "react";
import { combineTranscripts, diarizeTranscript, extractMedical, fetchHealth, fetchImportedAudio, fetchImportedSession, transcribeAudio, type FrequencyGroup, type ImportedSession, type SpeakerContainer, type VoiceSegment } from "./api";
import { ExtractionView } from "./components/ExtractionView";
import { TranscriptView } from "./components/TranscriptView";
import { useAudioRecorder } from "./hooks/useAudioRecorder";
import { useSpeechRecognition } from "./hooks/useSpeechRecognition";
import { LANGUAGES, speechLangFor } from "./languages";
import type { MedicalExtraction } from "./types";

const SAMPLE = `Good morning. So tell me, what brings you in today?
Doctor, I've had a high fever and a bad cough for the last three days, and I feel very weak.
I also have a sore throat and a mild headache.
Let me check. Your temperature is 101 degrees and your blood pressure is 130/85.
This looks like a viral throat infection.
I'll prescribe Paracetamol 500 mg twice daily after meals for 5 days.
I'm also giving you Azithromycin 250 mg once daily for 3 days.
Take plenty of fluids and rest. You're allergic to penicillin, correct? Yes.
Come back for a follow-up in 5 days if the fever doesn't settle.`;

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function App() {
  const [locale, setLocale] = useState<string>("en_IN");
  const speech = useSpeechRecognition(speechLangFor(locale));
  const recorder = useAudioRecorder();
  const [extraction, setExtraction] = useState<MedicalExtraction | null>(null);
  const [provider, setProvider] = useState<string>("");
  const [warning, setWarning] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [corrections, setCorrections] = useState<Record<string, string>>({});
  const [correctionNote, setCorrectionNote] = useState<string | null>(null);
  const [transcribing, setTranscribing] = useState(false);
  // True when the transcript came back already STT-corrected (dual-ASR merge),
  // so /api/extract should skip its own correction pass.
  const [preCorrected, setPreCorrected] = useState(false);
  // Dual-ASR stage transcripts (shown live).
  const [dual, setDual] = useState(false);
  const [regionalTranscript, setRegionalTranscript] = useState("");
  const [englishTranscript, setEnglishTranscript] = useState("");
  const [combining, setCombining] = useState(false);
  // Speaker labelling (Doctor:/Patient:) via the LLM.
  const [labelSpeakers, setLabelSpeakers] = useState(true);
  const [diarizing, setDiarizing] = useState(false);
  const [diarizationMode, setDiarizationMode] = useState<"ai" | "hybrid" | "acoustic">("acoustic");
  const [maxSpeakers, setMaxSpeakers] = useState<2 | 3>(3);
  const [recordingBlob, setRecordingBlob] = useState<Blob | null>(null);
  const [voiceSegments, setVoiceSegments] = useState<VoiceSegment[]>([]);
  const [frequencyGroups, setFrequencyGroups] = useState<FrequencyGroup[]>([]);
  const [speakerContainers, setSpeakerContainers] = useState<SpeakerContainer[]>([]);
  const [independentTranscription, setIndependentTranscription] = useState(false);
  const [playingSegment, setPlayingSegment] = useState<number | null>(null);
  const [importedSession, setImportedSession] = useState<ImportedSession | null>(null);
  const [importedAudioUrl, setImportedAudioUrl] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Discover whether the server is in dual-ASR mode.
  useEffect(() => {
    fetchHealth()
      .then((h) => setDual(Boolean(h.transcription?.dual)))
      .catch(() => setDual(false));
  }, []);

  // DoctorDiary can open VoiceRX with a persisted session. Load every stage,
  // the extraction, and the original audio into the normal VoiceRX controls.
  useEffect(() => {
    const sessionId = new URLSearchParams(window.location.search).get("session_id");
    if (!sessionId) return;
    let cancelled = false;
    (async () => {
      try {
        const session = await fetchImportedSession(sessionId);
        const audio = await fetchImportedAudio(session.audio_url);
        if (cancelled) return;
        setImportedSession(session);
        setImportedAudioUrl(URL.createObjectURL(audio));
        setRecordingBlob(audio);
        setLocale(session.locale || "en_IN");
        setRegionalTranscript(session.regional || "");
        setEnglishTranscript(session.english || "");
        setPreCorrected(true);
        setCorrections(session.corrections ?? {});
        setProvider(session.provider || "DoctorDiary / VoiceRX");
        setExtraction(session.extraction ?? null);
        const importedTranscript = session.final || session.transcript || session.english || session.regional || "";
        speech.setTranscript(importedTranscript);
        if (importedTranscript.trim()) {
          try {
            const diarized = await diarizeTranscript(importedTranscript, session.locale, "acoustic", audio, 3);
            if (cancelled) return;
            setVoiceSegments(diarized.segments ?? []);
            setFrequencyGroups(diarized.frequencyGroups ?? []);
            setSpeakerContainers(diarized.speakerContainers ?? []);
            setIndependentTranscription(diarized.independentTranscription ?? false);
            if (diarized.diarized.trim()) speech.setTranscript(diarized.diarized);
          } catch (e) {
            if (!cancelled) setError(e instanceof Error ? e.message : "Frequency speaker detection failed.");
          }
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Could not load the DoctorDiary session.");
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const transcript = speech.finalTranscript;
  const active = recorder.recording || speech.listening;
  // Dual-ASR only applies to non-English: for English there is no regional
  // language to merge, so we use a single English transcription.
  const dualActive = dual && locale !== "en_IN";

  // Starting the mic runs BOTH audio recording and live transcription together.
  async function handleStart() {
    setError(null);
    speech.start();
    await recorder.start();
  }

  // On stop, transcribe the recording. In dual-ASR mode we show the regional and
  // English transcripts as each arrives, then merge them into the final result.
  async function handleStop() {
    speech.stop();
    const blob = await recorder.stop();
    if (!blob) return;
    await processRecording(blob);
  }

  // Shared by handleStop (live mic recording) and handleFileSelected
  // (an existing audio file picked from disk) — both end up with a Blob to
  // transcribe and, optionally, diarize.
  async function processRecording(blob: Blob) {
    setRecordingBlob(blob);
    setError(null);
    setRegionalTranscript("");
    setEnglishTranscript("");

    if (dualActive) {
      setTranscribing(true);
      try {
        // Run both engines in parallel; surface each transcript as it resolves.
        const regionalP = transcribeAudio(blob, locale, "regional")
          .then((r) => {
            setRegionalTranscript(r.transcript);
            return r.transcript;
          })
          .catch((e) => {
            setError(`Regional STT failed: ${e instanceof Error ? e.message : e}`);
            return "";
          });
        const englishP = transcribeAudio(blob, locale, "english")
          .then((r) => {
            setEnglishTranscript(r.transcript);
            return r.transcript;
          })
          .catch((e) => {
            setError(`English STT failed: ${e instanceof Error ? e.message : e}`);
            return "";
          });
        const [reg, eng] = await Promise.all([regionalP, englishP]);
        if (!reg && !eng) {
          setError("Both transcriptions returned empty. Try recording again, a bit louder.");
          return;
        }
        // Merge: regional conversation + English medicine names (Groq).
        setCombining(true);
        const merged = await combineTranscripts(reg, eng, locale);
        setPreCorrected(true);
        setCorrections(merged.corrections ?? {});
        setCorrectionNote(merged.note ?? null);
        await finishTranscript(merged.corrected, blob);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Transcription failed.");
      } finally {
        setCombining(false);
        setTranscribing(false);
      }
      return;
    }

    // Single-provider mode.
    setTranscribing(true);
    try {
      const result = await transcribeAudio(blob, locale);
      if (result.transcript.trim()) {
        setPreCorrected(Boolean(result.corrected));
        if (result.corrections && Object.keys(result.corrections).length > 0) {
          setCorrections(result.corrections);
        }
        await finishTranscript(result.transcript.trim(), blob);
      } else if (!transcript.trim()) {
        setError("Transcription returned empty text. Try recording again, a bit louder.");
      }
    } catch (e) {
      setError(
        (e instanceof Error ? e.message : "Transcription failed.") +
          (transcript.trim() ? " Using the live preview transcript instead." : "")
      );
    } finally {
      setTranscribing(false);
    }
  }

  // Load an existing recording from disk (e.g. to test hybrid/acoustic
  // speaker detection against a real past conversation) instead of using
  // the mic. Runs through the exact same transcribe→diarize pipeline.
  async function handleFileSelected(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    if (active) {
      setError("Stop the current recording before loading a file.");
      return;
    }
    await processRecording(file);
  }

  async function handleExtract() {
    if (!transcript.trim()) {
      setError("There's no transcript to extract from yet.");
      return;
    }
    setBusy(true);
    setError(null);
    setWarning(null);
    if (!preCorrected) {
      setCorrections({});
      setCorrectionNote(null);
    }
    try {
      const res = await extractMedical(transcript, locale, preCorrected ? false : undefined);
      setExtraction(res.extraction);
      setProvider(res.provider);
      setWarning(res.warning ?? null);
      // When extract ran its own correction (non-dual), reflect its results.
      if (!preCorrected) {
        setCorrections(res.corrections ?? {});
        setCorrectionNote(res.correctionNote ?? null);
        if (res.correctedTranscript && res.correctedTranscript.trim() && res.correctedTranscript !== transcript) {
          speech.setTranscript(res.correctedTranscript);
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Extraction failed.");
    } finally {
      setBusy(false);
    }
  }

  // Manual edits / sample load mean the text is no longer the dual-corrected
  // output, so let /api/extract correct it again.
  function handleTranscriptChange(text: string) {
    speech.setTranscript(text);
    setPreCorrected(false);
  }

  // Show the final transcript, then optionally label speakers (Doctor:/Patient:).
  async function finishTranscript(finalText: string, audio?: Blob) {
    speech.setTranscript(finalText);
    if (labelSpeakers && finalText.trim()) {
      setDiarizing(true);
      try {
        const result = await diarizeTranscript(finalText, locale, diarizationMode, audio, maxSpeakers);
        setVoiceSegments(result.segments ?? []);
        setFrequencyGroups(result.frequencyGroups ?? []);
        setSpeakerContainers(result.speakerContainers ?? []);
        setIndependentTranscription(result.independentTranscription ?? false);
        if (result.diarized.trim()) speech.setTranscript(result.diarized);
      } catch {
        /* keep the unlabeled transcript on failure */
      } finally {
        setDiarizing(false);
      }
    }
  }

  // Label speakers on the current transcript on demand (e.g. after editing/paste).
  async function handleLabelSpeakers() {
    if (!transcript.trim()) return;
    if ((diarizationMode === "hybrid" || diarizationMode === "acoustic") && !recordingBlob) {
      setError("This speaker detection mode requires the original recording.");
      return;
    }
    setDiarizing(true);
    try {
      const result = await diarizeTranscript(transcript, locale, diarizationMode, recordingBlob ?? undefined, maxSpeakers);
      setVoiceSegments(result.segments ?? []);
      setFrequencyGroups(result.frequencyGroups ?? []);
      setSpeakerContainers(result.speakerContainers ?? []);
      setIndependentTranscription(result.independentTranscription ?? false);
      if (result.diarized.trim()) speech.setTranscript(result.diarized);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Diarization failed.");
    } finally {
      setDiarizing(false);
    }
  }

  function playVoiceSegment(index: number) {
    const segment = voiceSegments[index];
    const audio = audioRef.current;
    if (!segment || !audio) return;
    audio.currentTime = segment.start;
    setPlayingSegment(index);
    void audio.play().catch(() => setPlayingSegment(null));
  }

  function handleAudioTimeUpdate() {
    const index = playingSegment;
    const segment = index === null ? undefined : voiceSegments[index];
    const audio = audioRef.current;
    if (!segment || !audio || audio.currentTime < segment.end) return;
    audio.pause();
    audio.currentTime = segment.start;
    setPlayingSegment(null);
  }

  function handleClear() {
    speech.reset();
    recorder.reset();
    setExtraction(null);
    setProvider("");
    setWarning(null);
    setError(null);
    setCorrections({});
    setCorrectionNote(null);
    setPreCorrected(false);
    setRegionalTranscript("");
    setEnglishTranscript("");
    setCombining(false);
    setRecordingBlob(null);
    setVoiceSegments([]);
    setFrequencyGroups([]);
    setSpeakerContainers([]);
    setIndependentTranscription(false);
    setPlayingSegment(null);
    setImportedSession(null);
    if (importedAudioUrl) URL.revokeObjectURL(importedAudioUrl);
    setImportedAudioUrl(null);
  }

  const playbackUrl = recorder.audioUrl || importedAudioUrl;
  const showImportedStages = Boolean(importedSession);

  return (
    <div className="app">
      <header className="app-header">
        <h1>🎙️ VoiceRX</h1>
        <p>Record a doctor–patient conversation — Whisper transcribes the recording, then it extracts the clinical details.</p>
      </header>

      <main className="layout">
        <section className="panel">
          <div className="panel-head">
            <h2>Conversation</h2>
            <div className="controls">
              <select
                className="lang-select"
                value={locale}
                onChange={(e) => setLocale(e.target.value)}
                disabled={active || transcribing}
                aria-label="Spoken language"
                title="Language spoken in the consultation"
              >
                {LANGUAGES.map((l) => (
                  <option key={l.locale} value={l.locale}>
                    🌐 {l.label}
                  </option>
                ))}
              </select>

              {active ? (
                <button className="btn btn-stop" onClick={handleStop}>
                  ⏹ Stop
                  {recorder.recording && <span className="rec-dot" aria-hidden /> }
                  <span className="rec-time">{formatTime(recorder.elapsed)}</span>
                </button>
              ) : (
                <button className="btn btn-rec" onClick={handleStart}>
                  ● Record
                </button>
              )}

              <button className="btn btn-ghost" onClick={() => handleTranscriptChange(SAMPLE)}>
                Load sample
              </button>
              <label className="btn btn-ghost" style={{ cursor: "pointer" }}>
                📁 Upload audio
                <input
                  type="file"
                  accept="audio/*,.webm,.m4a,.ogg,.wav,.mp3"
                  onChange={handleFileSelected}
                  disabled={active || transcribing || diarizing}
                  style={{ display: "none" }}
                />
              </label>
              <button className="btn btn-ghost" onClick={handleClear}>
                Clear
              </button>
              <label className="toggle" title="Label each turn as Doctor or Patient">
                <input
                  type="checkbox"
                  checked={labelSpeakers}
                  onChange={(e) => setLabelSpeakers(e.target.checked)}
                  disabled={active}
                />
                👥 Label Doctor/Patient
              </label>
              <select
                className="lang-select"
                value={diarizationMode}
                onChange={(e) => setDiarizationMode(e.target.value as "ai" | "hybrid" | "acoustic")}
                disabled={active || transcribing || diarizing}
                aria-label="Speaker detection mode"
                title="Choose how Doctor and Patient speakers are distinguished"
              >
                <option value="ai">🤖 AI intent labeling</option>
                <option value="hybrid">🎙️ Voice frequency + AI</option>
                <option value="acoustic">🧠 Acoustic only (User 1/User 2, no AI)</option>
              </select>
              {(diarizationMode === "hybrid" || diarizationMode === "acoustic") && (
                <select
                  className="lang-select"
                  value={maxSpeakers}
                  onChange={(e) => setMaxSpeakers(Number(e.target.value) as 2 | 3)}
                  disabled={active || transcribing || diarizing}
                  aria-label="Maximum speakers"
                  title="Maximum number of distinct speakers to detect"
                >
                  <option value={2}>👤👤 2 speakers</option>
                  <option value={3}>👤👤👤 Up to 3 speakers</option>
                </select>
              )}
              <button
                className="btn btn-ghost"
                onClick={handleLabelSpeakers}
                disabled={diarizing || !transcript.trim() || active}
                title="Label speakers on the current transcript"
              >
                {diarizing ? "Labeling…" : "Label now"}
              </button>
            </div>
          </div>

          {!speech.supported && (
            <p className="msg msg-info">
              This browser has no live preview, but that's fine — the recording is
              transcribed accurately by Whisper when you press Stop. You can also type/paste.
            </p>
          )}
          {!recorder.supported && (
            <p className="msg msg-warn">Audio recording isn't supported in this browser.</p>
          )}

          {(showImportedStages || dualActive) && (showImportedStages || regionalTranscript || englishTranscript || transcribing) && (
            <div className="stages">
              <div className="stage">
                <div className="stage-head">
                  <span>🗣️ Regional (IndicConformer)</span>
                  {transcribing && !regionalTranscript && <span className="stage-spin">…</span>}
                </div>
                <div className="stage-body" dir="auto">
                  {regionalTranscript || (showImportedStages ? <span className="stage-hint">No result returned</span> : <span className="stage-hint">transcribing…</span>)}
                </div>
              </div>
              <div className="stage">
                <div className="stage-head">
                  <span>🔤 English (Whisper)</span>
                  {transcribing && !englishTranscript && <span className="stage-spin">…</span>}
                </div>
                <div className="stage-body" dir="auto">
                  {englishTranscript || (showImportedStages ? <span className="stage-hint">No result returned</span> : <span className="stage-hint">transcribing…</span>)}
                </div>
              </div>
              <div className="stage-merge">
                {combining
                  ? "⚙️ Merging with Groq — keeping the conversation, medicine names in English…"
                  : "⬇️ Final merged result below"}
              </div>
            </div>
          )}

          {(dualActive || showImportedStages) && <div className="final-label">✅ Final transcript (conversation + English medicine names)</div>}
          <TranscriptView
            value={transcript}
            interim={speech.interimTranscript}
            listening={speech.listening}
            onChange={handleTranscriptChange}
          />

          {playbackUrl && !recorder.recording && (
            <div className="playback">
              <span className="playback-label">🔊 Recording</span>
              <audio ref={audioRef} controls src={playbackUrl} onTimeUpdate={handleAudioTimeUpdate} onPause={() => setPlayingSegment(null)} />
              <a className="btn btn-ghost" href={playbackUrl} download="voicerx-recording.webm">
                ⬇ Download
              </a>
            </div>
          )}

          {voiceSegments.length > 0 && (
            <div className="voice-segments">
              <div className="voice-segments-head">
                <h3>Frequency voice groups</h3>
                <span>{frequencyGroups.length || new Set(voiceSegments.map((segment) => segment.speaker)).size} voices · {voiceSegments.length} segments</span>
              </div>
              {frequencyGroups.length > 0 && (
                <div className="frequency-group-list">
                  {frequencyGroups.map((group) => (
                    <span className="frequency-group" key={group.speaker}>
                      Speaker {group.speaker}: {Math.round(group.minHz)}–{Math.round(group.maxHz)} Hz
                    </span>
                  ))}
                </div>
              )}
              <div className="voice-segment-list">
                {voiceSegments.map((segment, index) => (
                  <div className={`voice-segment speaker-${segment.speaker.toLowerCase()}`} key={`${segment.speaker}-${segment.start}-${index}`}>
                    <b>Speaker {segment.speaker}</b>
                    {segment.text && <p className="voice-segment-text">{segment.text}</p>}
                    <span>{formatTime(Math.floor(segment.start))} – {formatTime(Math.ceil(segment.end))}</span>
                    <button className="voice-segment-play" onClick={() => playVoiceSegment(index)} disabled={!playbackUrl}>
                      {playingSegment === index ? "■ Stop" : "▶ Play"}
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {speakerContainers.length > 0 && (
            <div className="speaker-containers">
              <div className="speaker-containers-head">
                <h3>Per-speaker transcripts</h3>
                <div className="speaker-containers-meta">
                  <span>{speakerContainers.length} speaker{speakerContainers.length > 1 ? "s" : ""} · frequency analysis</span>
                  {independentTranscription
                    ? <span className="badge-independent">✅ Independent STT per speaker</span>
                    : <span className="badge-heuristic">⚡ Heuristic split</span>
                  }
                </div>
              </div>
              <div className="speaker-containers-grid">
                {speakerContainers.map((container) => (
                  <div
                    className={`speaker-container speaker-container-${container.speaker.toLowerCase()}`}
                    key={container.speaker}
                  >
                    <div className="speaker-container-header">
                      <div className="speaker-container-badge">
                        {container.speaker === "A" ? "🩺" : container.speaker === "B" ? "🧑" : "👥"}
                        <span className="speaker-container-label">{container.label}</span>
                        <span className="speaker-container-tag">Speaker {container.speaker}</span>
                      </div>
                      <span className="speaker-container-count">{container.chunks.length} turn{container.chunks.length !== 1 ? "s" : ""}</span>
                    </div>
                    <div className="speaker-container-body" dir="auto">
                      {container.chunks.length > 0 ? (
                        container.chunks.map((chunk, i) => (
                          <p className="speaker-chunk" key={i}>{chunk}</p>
                        ))
                      ) : (
                        <span className="stage-hint">No speech attributed to this speaker</span>
                      )}
                    </div>
                    {container.rawTranscript && container.rawTranscript !== container.transcript && (
                      <details className="speaker-raw-details">
                        <summary className="speaker-raw-summary">🔤 Raw STT output</summary>
                        <p className="speaker-raw-text" dir="auto">{container.rawTranscript}</p>
                      </details>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {speech.error && <p className="msg msg-warn">⚠️ {speech.error}</p>}
          {recorder.error && <p className="msg msg-error">{recorder.error}</p>}
          {transcribing && !dualActive && (
            <p className="msg msg-info">🎧 Transcribing the recording…</p>
          )}
          {diarizing && <p className="msg msg-info">👥 Labeling speakers (Doctor / Patient)…</p>}

          <button
            className="btn btn-primary btn-extract"
            onClick={handleExtract}
            disabled={busy || transcribing || active}
          >
            {busy ? "Extracting…" : "✨ Extract medical details"}
          </button>
        </section>

        <section className="panel">
          <div className="panel-head">
            <h2>Extracted details</h2>
            {provider && <span className="badge">engine: {provider}</span>}
          </div>

          {error && <p className="msg msg-error">{error}</p>}
          {warning && <p className="msg msg-warn">{warning}</p>}
          {correctionNote && <p className="msg msg-warn">{correctionNote}</p>}

          {Object.keys(corrections).length > 0 && (
            <div className="card card-wide corrections">
              <h3>🩹 Transcript corrections applied</h3>
              <ul className="corr-list">
                {Object.entries(corrections).map(([from, to], i) => (
                  <li key={`corr-${i}`}>
                    <span className="corr-from">{from}</span>
                    <span className="corr-arrow">→</span>
                    <b className="corr-to">{to}</b>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {extraction ? (
            <ExtractionView data={extraction} />
          ) : (
            <p className="placeholder">
              Record or paste a conversation, then click <b>Extract</b> to see symptoms,
              diagnoses, prescriptions, and follow-up details here.
            </p>
          )}
        </section>
      </main>

      <footer className="app-footer">
        <p>
          ⚠️ Documentation aid only — not medical advice. Output must be reviewed by a
          clinician. Handles PHI: secure & authenticate before any real-world use.
        </p>
      </footer>
    </div>
  );
}
