# Flutter integration

VoiceRX exposes two integration options:

## Option 1: WebView

Start VoiceRX on a machine reachable by the phone:

```bash
npm run start:all
```

Find that machine's LAN IP, for example `192.168.1.20`, then load this URL in
the Flutter WebView:

```text
http://192.168.1.20:5173/
```

The page uses the Vite proxy to call the API at port `4000`. The WebView needs
microphone permission. For Android development, add `INTERNET` permission and
allow cleartext HTTP; for iOS development, allow local-network HTTP traffic in
the app's ATS settings. Use HTTPS and a restricted `CLIENT_ORIGIN` in production.

## Option 2: Native Flutter API client

Use the API directly with a base URL such as:

```text
http://192.168.1.20:4000/api
```

### Health

```http
GET /health
```

### Transcription

Send a multipart request to `POST /transcribe`:

| Field | Type | Example |
|---|---|---|
| `audio` | file | `recording.m4a` |
| `locale` | text | `hi_IN` |
| `mode` | text, optional | `auto`, `regional`, or `english` |

Response:

```json
{
  "transcript": "…",
  "model": "whisper-large-v3",
  "provider": "groq"
}
```

### Speaker diarization

Send a multipart request to `POST /diarize`:

| Field | Type | Required |
|---|---|---|
| `audio` | file | Hybrid mode only |
| `transcript` | text | yes |
| `mode` | text | `ai` or `hybrid` |
| `locale` | text | optional |

`ai` uses the existing Groq intent-based Doctor/Patient labeling. `hybrid`
uses local speaker embeddings plus frequency features to produce Speaker A/B
segments, then Groq maps them to Doctor/Patient. Hybrid responses include:

```json
{
  "diarized": "Doctor: …\nPatient: …",
  "detectedVoices": 2,
  "segments": [
    { "speaker": "A", "start": 0.0, "end": 2.3 },
    { "speaker": "B", "start": 2.3, "end": 4.6 }
  ]
}
```

### Medical extraction

Send JSON to `POST /extract`:

```json
{
  "transcript": "Doctor: …\nPatient: …",
  "locale": "hi_IN",
  "correct": true
}
```

### Flutter multipart example

```dart
final request = http.MultipartRequest(
  'POST',
  Uri.parse('$baseUrl/transcribe'),
)
  ..fields['locale'] = 'hi_IN'
  ..fields['mode'] = 'auto'
  ..files.add(await http.MultipartFile.fromPath('audio', audioPath));

final response = await request.send();
final body = await response.stream.bytesToString();
```

Do not ship Groq keys in the Flutter app. Keep all provider credentials in the
VoiceRX server environment.
