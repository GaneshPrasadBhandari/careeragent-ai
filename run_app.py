from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    """
    Description: Launch FastAPI brain + Streamlit Mission Control locally (canonical supervisor).
    Layer: L0
    Input: None
    Output: exit code
    """
    root = Path(__file__).resolve().parent

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")

    api_cmd = [
        sys.executable, "-m", "uvicorn",
        "careeragent.api.main:app",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--reload",
    ]

    ui_cmd = [
        sys.executable, "-m", "streamlit",
        "run", "app/main.py",
        "--server.port", "8501",
    ]

    print("\n== CareerAgent-AI Local Launcher ==")
    print("API: http://127.0.0.1:8000/health  | docs: http://127.0.0.1:8000/docs")
    print("UI : http://localhost:8501\n")

    print("Starting FastAPI:", " ".join(api_cmd))
    api = subprocess.Popen(api_cmd, cwd=str(root), env=env)

    time.sleep(1.2)

    print("Starting Streamlit:", " ".join(ui_cmd))
    ui = subprocess.Popen(ui_cmd, cwd=str(root), env=env)

    try:
        while True:
            a = api.poll()
            u = ui.poll()

            # If one dies, stop the other and exit with that code
            if a is not None:
                print(f"FastAPI exited with code {a}. Stopping Streamlit…")
                ui.terminate()
                return a
            if u is not None:
                print(f"Streamlit exited with code {u}. Stopping FastAPI…")
                api.terminate()
                return u

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("Stopping…")
        api.terminate()
        ui.terminate()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())