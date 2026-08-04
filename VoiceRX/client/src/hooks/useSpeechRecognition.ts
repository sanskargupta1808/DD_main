import { useCallback, useEffect, useRef, useState } from "react";

export interface UseSpeechRecognition {
  /** Whether the browser supports the Web Speech API. */
  supported: boolean;
  listening: boolean;
  /** Finalised transcript accumulated so far. */
  finalTranscript: string;
  /** Current in-progress (not yet final) chunk. */
  interimTranscript: string;
  error: string | null;
  start: () => void;
  stop: () => void;
  reset: () => void;
  /** Replace the final transcript (e.g. after manual edits or file upload). */
  setTranscript: (text: string) => void;
}

export function useSpeechRecognition(lang = "en-US"): UseSpeechRecognition {
  const Ctor =
    typeof window !== "undefined"
      ? window.SpeechRecognition ?? window.webkitSpeechRecognition
      : undefined;
  const supported = Boolean(Ctor);

  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const listeningRef = useRef(false);
  const finalRef = useRef("");

  const [listening, setListening] = useState(false);
  const [finalTranscript, setFinalTranscript] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [error, setError] = useState<string | null>(null);

  const start = useCallback(() => {
    if (!Ctor || listeningRef.current) return;
    setError(null);

    const recognition = new Ctor();
    recognition.lang = lang;
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        const text = result[0].transcript;
        if (result.isFinal) {
          finalRef.current = (finalRef.current + " " + text).trim();
          setFinalTranscript(finalRef.current);
        } else {
          interim += text;
        }
      }
      setInterimTranscript(interim);
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      if (event.error === "no-speech" || event.error === "aborted") return;
      // These are fatal for the live preview — stop retrying so we don't loop,
      // and never block recording (Whisper transcribes the audio on Stop anyway).
      listeningRef.current = false;
      const messages: Record<string, string> = {
        "not-allowed": "Microphone permission is blocked for the live preview.",
        "service-not-allowed":
          "Live preview isn't available in this browser (its speech service is blocked). " +
          "Recording still works — the transcript is generated when you press Stop. " +
          "For a live preview, use Chrome/Edge on localhost and allow microphone access.",
        "audio-capture": "No microphone was found for the live preview.",
        "network": "Live preview needs a network connection and is unavailable right now.",
      };
      setError(messages[event.error] ?? `Live preview error: ${event.error}`);
    };

    recognition.onend = () => {
      // Auto-restart while the user still wants to listen (Chrome stops periodically).
      if (listeningRef.current) {
        try {
          recognition.start();
        } catch {
          /* ignore */
        }
      } else {
        setListening(false);
        setInterimTranscript("");
      }
    };

    recognitionRef.current = recognition;
    listeningRef.current = true;
    setListening(true);
    recognition.start();
  }, [Ctor, lang]);

  const stop = useCallback(() => {
    listeningRef.current = false;
    setListening(false);
    recognitionRef.current?.stop();
  }, []);

  const reset = useCallback(() => {
    finalRef.current = "";
    setFinalTranscript("");
    setInterimTranscript("");
    setError(null);
  }, []);

  const setTranscript = useCallback((text: string) => {
    finalRef.current = text;
    setFinalTranscript(text);
    setInterimTranscript("");
  }, []);

  useEffect(() => {
    return () => {
      listeningRef.current = false;
      recognitionRef.current?.abort();
    };
  }, []);

  return {
    supported,
    listening,
    finalTranscript,
    interimTranscript,
    error,
    start,
    stop,
    reset,
    setTranscript,
  };
}
