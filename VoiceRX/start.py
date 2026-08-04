#!/usr/bin/env python3
"""Cross-platform replacement for start.sh — works on macOS (Intel or Apple
Silicon), Linux, and Windows without needing bash/WSL/Git Bash.

Starts medicine_pipeline + indic_stt (Python sidecars), waits for both to
report healthy, then runs the Node/React dev servers in the foreground.
Ctrl+C stops everything.

Usage: python3 start.py   (or `python start.py` on Windows)
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

ROOT_DIR = Path(__file__).resolve().parent
IS_WINDOWS = sys.platform == "win32"


def venv_python(venv_dir: Path) -> Path:
    """Path to a venv's Python interpreter, on whichever OS this is."""
    return venv_dir / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


def load_env_file(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE per line, '#' comments) — avoids
    requiring python-dotenv to be installed just to run this launcher."""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def wait_for_health(name: str, url: str, attempts: int = 120) -> None:
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status < 400:
                    print(f"{name} is ready")
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(1)
    fail(f"{name} did not become healthy: {url}")


def main() -> None:
    # Without this, progress messages sit in Python's block buffer and never
    # appear (in a piped log or a non-interactive terminal) until exit.
    sys.stdout.reconfigure(line_buffering=True)
    os.chdir(ROOT_DIR)

    env_file = ROOT_DIR / ".env"
    if not env_file.is_file():
        fail("Missing .env. Copy .env.example to .env, add GROQ_API_KEY, then run this again.")

    indic_python = venv_python(ROOT_DIR / "indic_stt" / ".venv")
    if not indic_python.is_file():
        fail("Missing indic_stt/.venv. Install indic_stt/requirements.txt first.")

    medicine_python = venv_python(ROOT_DIR / "medicine_pipeline" / ".venv")
    if not medicine_python.is_file():
        fail("Missing medicine_pipeline/.venv. Install medicine_pipeline/requirements.txt first.")

    medicine_output = ROOT_DIR / "medicine_pipeline" / "output"
    required_artifacts = ["medicine.index", "medicine_meta.json", "bktree.pkl"]
    if not all((medicine_output / name).is_file() for name in required_artifacts):
        fail("Medicine search artifacts are missing from medicine_pipeline/output/.")

    if shutil.which("ffmpeg") is None:
        fail("ffmpeg is required for audio conversion but was not found on PATH.")

    npm = shutil.which("npm")
    if npm is None:
        fail("npm is required to run the VoiceRX Node/React dev servers but was not found on PATH.")

    load_env_file(env_file)
    os.environ.setdefault("INDIC_PRECISION", "fp32")
    os.environ.setdefault("INDIC_DECODING", "ctc")

    children: list[subprocess.Popen] = []

    def cleanup(*_args) -> None:
        for proc in children:
            if proc.poll() is None:
                proc.terminate()
        for proc in children:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    signal.signal(signal.SIGINT, lambda *_: (cleanup(), sys.exit(0)))
    if not IS_WINDOWS:
        signal.signal(signal.SIGTERM, lambda *_: (cleanup(), sys.exit(0)))

    try:
        print("Starting medicine search on http://127.0.0.1:8000 …")
        medicine_env = {**os.environ, "PORT": "8000", "HOST": "127.0.0.1"}
        children.append(subprocess.Popen(
            [str(medicine_python), "server.py"],
            cwd=ROOT_DIR / "medicine_pipeline",
            env=medicine_env,
        ))

        print("Starting IndicConformer "
              f"({os.environ['INDIC_PRECISION']}/{os.environ['INDIC_DECODING']}) "
              "on http://127.0.0.1:8001 …")
        indic_env = {**os.environ, "PORT": "8001"}
        children.append(subprocess.Popen(
            [str(indic_python), "-m", "uvicorn", "server:app",
             "--app-dir", "indic_stt", "--host", "127.0.0.1", "--port", "8001"],
            cwd=ROOT_DIR,
            env=indic_env,
        ))

        wait_for_health("Medicine search", "http://127.0.0.1:8000/health")
        wait_for_health("IndicConformer", "http://127.0.0.1:8001/health")

        print("Starting VoiceRX API and frontend …")
        node_proc = subprocess.Popen([npm, "run", "dev"], cwd=ROOT_DIR, env=os.environ.copy())
        children.append(node_proc)
        node_proc.wait()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
