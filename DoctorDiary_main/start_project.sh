#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VOICE_RX_DIR="${VOICERX_DIR:-/Users/LocalDownloads/Main/VoiceRX}"
VOICE_RX_API="http://127.0.0.1:${VOICERX_PORT:-5005}"
VOICE_RX_NODE_API="http://127.0.0.1:${VOICERX_NODE_PORT:-4000}"
VOICE_RX_FRONTEND="http://127.0.0.1:${VOICERX_FRONTEND_PORT:-5173}"
DOCTOR_DIARY_URL="http://127.0.0.1:${DD_PORT:-8080}"
LOG_DIR="${PROJECT_DIR}/.run-logs"
mkdir -p "$LOG_DIR"

child_pids=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${child_pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${child_pids[@]:-}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

wait_for_health() {
  local name="$1" url="$2" attempts=0
  until curl --silent --show-error --fail --max-time 3 "$url" >/dev/null 2>&1; do
    attempts=$((attempts + 1))
    if (( attempts >= 60 )); then
      echo "$name did not become ready: $url" >&2
      return 1
    fi
    sleep 1
  done
  echo "$name is ready"
}

if [[ ! -d "$VOICE_RX_DIR/flask_api" ]]; then
  echo "VoiceRX project not found at $VOICE_RX_DIR" >&2
  exit 1
fi

# Prefer VoiceRX's own environment. If that environment has a stale Python
# symlink, keep a persistent fallback inside DoctorDiary instead of relying on
# a temporary directory that disappears between sessions.
VOICE_RX_PYTHON="${VOICERX_PYTHON:-$VOICE_RX_DIR/flask_api/.venv/bin/python}"
if [[ ! -x "$VOICE_RX_PYTHON" ]] || ! "$VOICE_RX_PYTHON" -c 'import flask, flask_cors, dotenv' >/dev/null 2>&1; then
  LOCAL_VOICERX_VENV="$PROJECT_DIR/.voicerx-venv"
  LOCAL_VOICERX_PYTHON="$LOCAL_VOICERX_VENV/bin/python"
  if [[ ! -x "$LOCAL_VOICERX_PYTHON" ]]; then
    PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-$(command -v python3.12 || command -v python3 || true)}"
    if [[ -z "$PYTHON_BOOTSTRAP" ]]; then
      echo "Python 3.12 or Python 3 is required to prepare VoiceRX." >&2
      exit 1
    fi
    echo "Preparing persistent VoiceRX runtime at $LOCAL_VOICERX_VENV"
    "$PYTHON_BOOTSTRAP" -m venv "$LOCAL_VOICERX_VENV"
  fi
  if ! "$LOCAL_VOICERX_PYTHON" -c 'import flask, flask_cors, dotenv' >/dev/null 2>&1; then
    echo "Installing VoiceRX Flask dependencies (first run only)"
    "$LOCAL_VOICERX_PYTHON" -m pip install -r "$VOICE_RX_DIR/flask_api/requirements.txt"
  fi
  VOICE_RX_PYTHON="$LOCAL_VOICERX_PYTHON"
fi

if ! curl --silent --show-error --fail --max-time 2 "$VOICE_RX_API/api/health" >/dev/null 2>&1; then
  echo "Starting VoiceRX Flask API on ${VOICE_RX_API}"
  (
    cd "$VOICE_RX_DIR/flask_api"
    FLASK_PORT="${VOICERX_PORT:-5005}" exec "$VOICE_RX_PYTHON" app.py
  ) >"$LOG_DIR/voicerx-api.log" 2>&1 &
  child_pids+=("$!")
fi
wait_for_health "VoiceRX API" "$VOICE_RX_API/api/health"

if [[ ! -d "$VOICE_RX_DIR/client" ]]; then
  echo "VoiceRX frontend not found at $VOICE_RX_DIR/client" >&2
  exit 1
fi
if ! curl --silent --show-error --fail --max-time 2 "$VOICE_RX_FRONTEND" >/dev/null 2>&1; then
  echo "Starting VoiceRX frontend on ${VOICE_RX_FRONTEND}"
  (
    cd "$VOICE_RX_DIR"
    VITE_API_TARGET="http://127.0.0.1:${VOICERX_NODE_PORT:-4000}" \
      exec npm run dev -w client -- --host 127.0.0.1 --port "${VOICERX_FRONTEND_PORT:-5173}"
  ) >"$LOG_DIR/voicerx-frontend.log" 2>&1 &
  child_pids+=("$!")
fi
wait_for_health "VoiceRX frontend" "$VOICE_RX_FRONTEND"

# Start the supporting services when their configured environments are usable.
# The Flask API can still start in English/groq mode without them.
if [[ "${START_VOICERX_SUPPORTING_SERVICES:-auto}" != "never" ]]; then
  if [[ -x "$VOICE_RX_DIR/medicine_pipeline/.venv/bin/python" ]] && "$VOICE_RX_DIR/medicine_pipeline/.venv/bin/python" -c 'import fastapi, uvicorn' >/dev/null 2>&1; then
    if ! curl --silent --show-error --fail --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
      echo "Starting VoiceRX medicine search API on http://127.0.0.1:8000"
      (cd "$VOICE_RX_DIR/medicine_pipeline" && PORT=8000 HOST=127.0.0.1 exec .venv/bin/python server.py) >"$LOG_DIR/voicerx-medicine.log" 2>&1 &
      child_pids+=("$!")
      wait_for_health "VoiceRX medicine search" "http://127.0.0.1:8000/health"
    fi
  else
    echo "VoiceRX medicine service skipped: its Python environment is unavailable"
  fi

  if [[ -x "$VOICE_RX_DIR/indic_stt/.venv/bin/python" ]] && "$VOICE_RX_DIR/indic_stt/.venv/bin/python" -c 'import fastapi, uvicorn' >/dev/null 2>&1; then
    if ! curl --silent --show-error --fail --max-time 2 http://127.0.0.1:8001/health >/dev/null 2>&1; then
      echo "Starting VoiceRX Indic STT API on http://127.0.0.1:8001"
      (cd "$VOICE_RX_DIR" && PORT=8001 exec indic_stt/.venv/bin/python -m uvicorn server:app --app-dir indic_stt --host 127.0.0.1 --port 8001) >"$LOG_DIR/voicerx-indic-stt.log" 2>&1 &
      child_pids+=("$!")
      wait_for_health "VoiceRX Indic STT" "http://127.0.0.1:8001/health"
    fi
  else
    echo "VoiceRX Indic STT skipped: its Python environment is unavailable"
  fi
fi

# The React frontend proxies /api requests to this Node service. Keep it
# separate from the Flask integration API above: diarization, transcription,
# extraction, and the updated frequency-speaker route are served here.
if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to start the VoiceRX Node API" >&2
  exit 1
fi
if [[ ! -x "$VOICE_RX_DIR/node_modules/.bin/tsx" ]]; then
  echo "VoiceRX Node dependencies are missing. Run 'npm install' in $VOICE_RX_DIR first." >&2
  exit 1
fi
if ! curl --silent --show-error --fail --max-time 2 "$VOICE_RX_NODE_API/api/health" >/dev/null 2>&1; then
  echo "Starting VoiceRX Node API on ${VOICE_RX_NODE_API}"
  (
    cd "$VOICE_RX_DIR"
    HOST=127.0.0.1 \
      PORT="${VOICERX_NODE_PORT:-4000}" \
      TRANSCRIPTION_BASE_URL="http://127.0.0.1:8001" \
      MEDICINE_SEARCH_URL="http://127.0.0.1:8000" \
      exec npm run dev -w server
  ) >"$LOG_DIR/voicerx-node.log" 2>&1 &
  child_pids+=("$!")
fi
wait_for_health "VoiceRX Node API" "$VOICE_RX_NODE_API/api/health"

echo "Starting DoctorDiary on ${DOCTOR_DIARY_URL}"
(cd "$PROJECT_DIR" && DD_HOST=127.0.0.1 DD_PORT="${DD_PORT:-8080}" exec python3 server.py) >"$LOG_DIR/doctor-diary.log" 2>&1 &
child_pids+=("$!")
wait_for_health "DoctorDiary" "$DOCTOR_DIARY_URL/api/dashboard-stats"

echo
echo "DoctorDiary: ${DOCTOR_DIARY_URL}"
echo "VoiceRX:    ${VOICE_RX_API}"
echo "VoiceRX API: ${VOICE_RX_NODE_API}"
echo "VoiceRX UI: ${VOICE_RX_FRONTEND}"
echo "Logs:       ${LOG_DIR}"
echo "Press Ctrl+C to stop all services."
wait
