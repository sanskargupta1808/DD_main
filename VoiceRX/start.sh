#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f .env ]]; then
  echo "Missing .env. Copy .env.example to .env, add GROQ_API_KEY, then run ./start.sh again."
  exit 1
fi

if [[ ! -x indic_stt/.venv/bin/python ]]; then
  echo "Missing indic_stt/.venv. Install indic_stt/requirements.txt first."
  exit 1
fi

if [[ ! -x medicine_pipeline/.venv/bin/python ]]; then
  echo "Missing medicine_pipeline/.venv. Install medicine_pipeline/requirements.txt first."
  exit 1
fi

if [[ ! -f medicine_pipeline/output/medicine.index ||
      ! -f medicine_pipeline/output/medicine_meta.json ||
      ! -f medicine_pipeline/output/bktree.pkl ]]; then
  echo "Medicine search artifacts are missing from medicine_pipeline/output/."
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required for audio conversion but was not found."
  exit 1
fi

# Load local configuration without printing secrets.
set -a
# shellcheck disable=SC1091
source .env
set +a

export INDIC_PRECISION="${INDIC_PRECISION:-fp32}"
export INDIC_DECODING="${INDIC_DECODING:-ctc}"

child_pids=()

cleanup() {
  trap - EXIT INT TERM
  for pid in "${child_pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${child_pids[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

wait_for_health() {
  local name="$1"
  local url="$2"
  local attempts=0
  until curl --silent --show-error --fail --max-time 3 "$url" >/dev/null 2>&1; do
    attempts=$((attempts + 1))
    if (( attempts >= 120 )); then
      echo "$name did not become healthy: $url"
      exit 1
    fi
    sleep 1
  done
  echo "$name is ready"
}

echo "Starting medicine search on http://127.0.0.1:8000 …"
(
  cd medicine_pipeline
  PORT=8000 HOST=127.0.0.1 exec .venv/bin/python server.py
) &
child_pids+=("$!")

echo "Starting IndicConformer ($INDIC_PRECISION/$INDIC_DECODING) on http://127.0.0.1:8001 …"
(
  PORT=8001 exec indic_stt/.venv/bin/python -m uvicorn server:app \
    --app-dir indic_stt --host 127.0.0.1 --port 8001
) &
child_pids+=("$!")

wait_for_health "Medicine search" "http://127.0.0.1:8000/health"
wait_for_health "IndicConformer" "http://127.0.0.1:8001/health"

echo "Starting VoiceRX API and frontend …"
npm run dev
