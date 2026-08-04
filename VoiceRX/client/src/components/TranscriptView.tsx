interface Props {
  value: string;
  interim: string;
  listening: boolean;
  onChange: (text: string) => void;
}

export function TranscriptView({ value, interim, listening, onChange }: Props) {
  return (
    <div className="transcript">
      <textarea
        className="transcript-area"
        value={value}
        placeholder="Start recording, upload an audio file, or type/paste the conversation here…"
        onChange={(e) => onChange(e.target.value)}
        rows={10}
        aria-label="Conversation transcript"
      />
      {listening && (
        <div className="interim" aria-live="polite">
          {interim ? interim : <span className="interim-hint">listening…</span>}
        </div>
      )}
    </div>
  );
}
