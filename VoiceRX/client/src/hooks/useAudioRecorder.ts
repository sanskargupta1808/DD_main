import { useCallback, useEffect, useRef, useState } from "react";

export interface UseAudioRecorder {
  /** Whether MediaRecorder is supported by this browser. */
  supported: boolean;
  recording: boolean;
  /** Object URL of the last completed recording (for playback/download). */
  audioUrl: string | null;
  /** Elapsed recording time in seconds. */
  elapsed: number;
  error: string | null;
  start: () => Promise<void>;
  /** Stops recording and resolves with the recorded audio Blob (or null). */
  stop: () => Promise<Blob | null>;
  reset: () => void;
}

function pickMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined") return undefined;
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"];
  return candidates.find((t) => MediaRecorder.isTypeSupported(t));
}

export function useAudioRecorder(): UseAudioRecorder {
  const supported =
    typeof window !== "undefined" &&
    typeof MediaRecorder !== "undefined" &&
    !!navigator.mediaDevices?.getUserMedia;

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const urlRef = useRef<string | null>(null);
  const stopResolverRef = useRef<((blob: Blob | null) => void) | null>(null);

  const [recording, setRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const clearTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  const revokeUrl = () => {
    if (urlRef.current) {
      URL.revokeObjectURL(urlRef.current);
      urlRef.current = null;
    }
  };

  const start = useCallback(async () => {
    if (!supported || recorderRef.current) return;
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];
      revokeUrl();
      setAudioUrl(null);
      setElapsed(0);

      const mimeType = pickMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        });
        const url = URL.createObjectURL(blob);
        urlRef.current = url;
        setAudioUrl(url);
        streamRef.current?.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
        recorderRef.current = null;
        // Hand the finished recording to whoever awaited stop().
        if (stopResolverRef.current) {
          stopResolverRef.current(blob.size > 0 ? blob : null);
          stopResolverRef.current = null;
        }
      };

      recorderRef.current = recorder;
      recorder.start();
      setRecording(true);

      const startedAt = Date.now();
      timerRef.current = setInterval(() => {
        setElapsed(Math.floor((Date.now() - startedAt) / 1000));
      }, 250);
    } catch (e) {
      setError(
        e instanceof Error && e.name === "NotAllowedError"
          ? "Microphone access was denied."
          : "Could not start audio recording."
      );
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      recorderRef.current = null;
    }
  }, [supported]);

  const stop = useCallback((): Promise<Blob | null> => {
    clearTimer();
    setRecording(false);
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive") {
      return Promise.resolve(null);
    }
    return new Promise<Blob | null>((resolve) => {
      stopResolverRef.current = resolve;
      recorder.stop();
    });
  }, []);

  const reset = useCallback(() => {
    revokeUrl();
    setAudioUrl(null);
    setElapsed(0);
  }, []);

  useEffect(() => {
    return () => {
      clearTimer();
      revokeUrl();
      streamRef.current?.getTracks().forEach((t) => t.stop());
      const recorder = recorderRef.current;
      if (recorder && recorder.state !== "inactive") recorder.stop();
    };
  }, []);

  return { supported, recording, audioUrl, elapsed, error, start, stop, reset };
}
