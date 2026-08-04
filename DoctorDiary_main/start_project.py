#!/usr/bin/env python3
"""Cross-platform replacement for start_project.sh — works on macOS (Intel or
Apple Silicon), Linux, and Windows without needing bash/WSL/Git Bash.

Starts VoiceRX's Flask API + frontend + (if usable) its Python sidecars +
Node API, then DoctorDiary itself, waiting for each to report healthy.
Ctrl+C stops everything this script started.

Usage: python3 start_project.py   (or `python start_project.py` on Windows)

Env vars (all optional):
    VOICERX_DIR            default: the sibling "VoiceRX" directory next to
                            this project (override if it lives elsewhere)
    VOICERX_PYTHON          path to a Python with flask/flask_cors/dotenv
                            installed (default: VoiceRX's own flask_api venv,
                            falling back to a local .voicerx-venv here)
    VOICERX_PORT            default 5005
    VOICERX_NODE_PORT       default 4000
    VOICERX_FRONTEND_PORT   default 5173
    DD_HOST / DD_PORT       default 127.0.0.1 / 8080
    START_VOICERX_SUPPORTING_SERVICES   set to "never" to skip medicine
                            search + IndicConformer STT even if usable
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
IS_WINDOWS = sys.platform == "win32"

VOICE_RX_DIR = Path(os.environ.get("VOICERX_DIR", str(PROJECT_DIR.parent / "VoiceRX")))
VOICE_RX_PORT = os.environ.get("VOICERX_PORT", "5005")
VOICE_RX_NODE_PORT = os.environ.get("VOICERX_NODE_PORT", "4000")
VOICE_RX_FRONTEND_PORT = os.environ.get("VOICERX_FRONTEND_PORT", "5173")
DD_HOST = os.environ.get("DD_HOST", "127.0.0.1")
DD_PORT = os.environ.get("DD_PORT", "8080")

VOICE_RX_API = f"http://127.0.0.1:{VOICE_RX_PORT}"
VOICE_RX_NODE_API = f"http://127.0.0.1:{VOICE_RX_NODE_PORT}"
VOICE_RX_FRONTEND = f"http://127.0.0.1:{VOICE_RX_FRONTEND_PORT}"
DOCTOR_DIARY_URL = f"http://127.0.0.1:{DD_PORT}"

LOG_DIR = PROJECT_DIR / ".run-logs"
LOG_DIR.mkdir(exist_ok=True)

children: list[subprocess.Popen] = []
log_files: list = []


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def python_has(python: Path, *modules: str) -> bool:
    if not python.is_file():
        return False
    code = "; ".join(f"import {m}" for m in modules)
    return subprocess.run([str(python), "-c", code], capture_output=True).returncode == 0


def is_healthy(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status < 400
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return False


def wait_for_health(name: str, url: str, attempts: int = 60) -> bool:
    for _ in range(attempts):
        if is_healthy(url, timeout=3):
            print(f"{name} is ready")
            return True
        time.sleep(1)
    print(f"{name} did not become ready: {url}", file=sys.stderr)
    return False


def spawn(name: str, cmd: list[str], cwd: Path, env: dict | None = None) -> subprocess.Popen:
    log_path = LOG_DIR / f"{name}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    log_files.append(log_file)
    proc = subprocess.Popen(cmd, cwd=cwd, env=env or os.environ.copy(),
                             stdout=log_file, stderr=subprocess.STDOUT)
    children.append(proc)
    return proc


def cleanup(*_args) -> None:
    for proc in children:
        if proc.poll() is None:
            proc.terminate()
    for proc in children:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    for f in log_files:
        f.close()


def resolve_voicerx_python() -> Path:
    """Prefer VoiceRX's own flask_api venv; fall back to a persistent local
    venv inside DoctorDiary if that one is missing/broken (e.g. a stale
    symlink) — mirrors start_project.sh's original fallback behavior."""
    override = os.environ.get("VOICERX_PYTHON")
    candidate = Path(override) if override else VOICE_RX_DIR / "flask_api" / ".venv" / (
        "Scripts/python.exe" if IS_WINDOWS else "bin/python"
    )
    if python_has(candidate, "flask", "flask_cors", "dotenv"):
        return candidate

    local_venv = PROJECT_DIR / ".voicerx-venv"
    local_python = venv_python(local_venv)
    if not python_has(local_python, "flask", "flask_cors", "dotenv"):
        bootstrap = (
            os.environ.get("PYTHON_BOOTSTRAP")
            or shutil.which("python3.12")
            or shutil.which("python3")
            or shutil.which("python")
        )
        if not bootstrap:
            fail("Python 3.12 or Python 3 is required to prepare VoiceRX.")
        print(f"Preparing persistent VoiceRX runtime at {local_venv}")
        subprocess.run([bootstrap, "-m", "venv", str(local_venv)], check=True)
        if not python_has(local_python, "flask", "flask_cors", "dotenv"):
            print("Installing VoiceRX Flask dependencies (first run only)")
            subprocess.run(
                [str(local_python), "-m", "pip", "install", "-r",
                 str(VOICE_RX_DIR / "flask_api" / "requirements.txt")],
                check=True,
            )
    return local_python


def main() -> None:
    # Without this, progress messages sit in Python's block buffer and never
    # appear (in a piped log or a non-interactive terminal) until exit.
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGINT, lambda *_: (cleanup(), sys.exit(0)))
    if not IS_WINDOWS:
        signal.signal(signal.SIGTERM, lambda *_: (cleanup(), sys.exit(0)))

    try:
        if not (VOICE_RX_DIR / "flask_api").is_dir():
            fail(f"VoiceRX project not found at {VOICE_RX_DIR} "
                 "(set VOICERX_DIR if it lives somewhere else)")

        voicerx_python = resolve_voicerx_python()

        if not is_healthy(f"{VOICE_RX_API}/api/health"):
            print(f"Starting VoiceRX Flask API on {VOICE_RX_API}")
            spawn("voicerx-api", [str(voicerx_python), "app.py"],
                  cwd=VOICE_RX_DIR / "flask_api",
                  env={**os.environ, "FLASK_PORT": VOICE_RX_PORT})
        if not wait_for_health("VoiceRX API", f"{VOICE_RX_API}/api/health"):
            fail("VoiceRX API failed to start.")

        if not (VOICE_RX_DIR / "client").is_dir():
            fail(f"VoiceRX frontend not found at {VOICE_RX_DIR / 'client'}")
        npm = shutil.which("npm")
        if not npm:
            fail("npm is required to start the VoiceRX frontend and Node API.")
        if not is_healthy(VOICE_RX_FRONTEND):
            print(f"Starting VoiceRX frontend on {VOICE_RX_FRONTEND}")
            spawn("voicerx-frontend",
                  [npm, "run", "dev", "-w", "client", "--",
                   "--host", "127.0.0.1", "--port", VOICE_RX_FRONTEND_PORT],
                  cwd=VOICE_RX_DIR,
                  env={**os.environ, "VITE_API_TARGET": f"http://127.0.0.1:{VOICE_RX_NODE_PORT}"})
        if not wait_for_health("VoiceRX frontend", VOICE_RX_FRONTEND):
            fail("VoiceRX frontend failed to start.")

        if os.environ.get("START_VOICERX_SUPPORTING_SERVICES", "auto") != "never":
            medicine_python = venv_python(VOICE_RX_DIR / "medicine_pipeline" / ".venv")
            if python_has(medicine_python, "fastapi", "uvicorn"):
                if not is_healthy("http://127.0.0.1:8000/health"):
                    print("Starting VoiceRX medicine search API on http://127.0.0.1:8000")
                    spawn("voicerx-medicine", [str(medicine_python), "server.py"],
                          cwd=VOICE_RX_DIR / "medicine_pipeline",
                          env={**os.environ, "PORT": "8000", "HOST": "127.0.0.1"})
                    wait_for_health("VoiceRX medicine search", "http://127.0.0.1:8000/health")
            else:
                print("VoiceRX medicine service skipped: its Python environment is unavailable")

            indic_python = venv_python(VOICE_RX_DIR / "indic_stt" / ".venv")
            if python_has(indic_python, "fastapi", "uvicorn"):
                if not is_healthy("http://127.0.0.1:8001/health"):
                    print("Starting VoiceRX Indic STT API on http://127.0.0.1:8001")
                    spawn("voicerx-indic-stt",
                          [str(indic_python), "-m", "uvicorn", "server:app",
                           "--app-dir", "indic_stt", "--host", "127.0.0.1", "--port", "8001"],
                          cwd=VOICE_RX_DIR,
                          env={**os.environ, "PORT": "8001"})
                    wait_for_health("VoiceRX Indic STT", "http://127.0.0.1:8001/health")
            else:
                print("VoiceRX Indic STT skipped: its Python environment is unavailable")

        # The React frontend proxies /api requests to this Node service. Kept
        # separate from the Flask integration API above: diarization,
        # transcription, extraction, and the frequency-speaker route live here.
        if not (VOICE_RX_DIR / "node_modules" / ".bin" / ("tsx.cmd" if IS_WINDOWS else "tsx")).is_file():
            fail(f"VoiceRX Node dependencies are missing. Run 'npm install' in {VOICE_RX_DIR} first.")
        if not is_healthy(f"{VOICE_RX_NODE_API}/api/health"):
            print(f"Starting VoiceRX Node API on {VOICE_RX_NODE_API}")
            spawn("voicerx-node", [npm, "run", "dev", "-w", "server"],
                  cwd=VOICE_RX_DIR,
                  env={
                      **os.environ,
                      "HOST": "127.0.0.1",
                      "PORT": VOICE_RX_NODE_PORT,
                      "TRANSCRIPTION_BASE_URL": "http://127.0.0.1:8001",
                      "MEDICINE_SEARCH_URL": "http://127.0.0.1:8000",
                  })
        if not wait_for_health("VoiceRX Node API", f"{VOICE_RX_NODE_API}/api/health"):
            fail("VoiceRX Node API failed to start.")

        if not is_healthy(f"{DOCTOR_DIARY_URL}/api/dashboard-stats"):
            print(f"Starting DoctorDiary on {DOCTOR_DIARY_URL}")
            spawn("doctor-diary", [sys.executable, "server.py"],
                  cwd=PROJECT_DIR,
                  env={**os.environ, "DD_HOST": DD_HOST, "DD_PORT": DD_PORT})
        if not wait_for_health("DoctorDiary", f"{DOCTOR_DIARY_URL}/api/dashboard-stats"):
            fail("DoctorDiary failed to start.")

        print()
        print(f"DoctorDiary: {DOCTOR_DIARY_URL}")
        print(f"VoiceRX:    {VOICE_RX_API}")
        print(f"VoiceRX API: {VOICE_RX_NODE_API}")
        print(f"VoiceRX UI: {VOICE_RX_FRONTEND}")
        print(f"Logs:       {LOG_DIR}")
        print("Press Ctrl+C to stop all services.")

        while True:
            time.sleep(1)
            for proc in children:
                if proc.poll() is not None:
                    fail(f"A service exited unexpectedly (see {LOG_DIR}). Stopping.")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
