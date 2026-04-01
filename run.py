"""
Convenience launcher — starts both services in parallel sub-processes.

Usage:
    python3 run.py

Empathy Engine  →  http://localhost:8001
Pitch Visualizer →  http://localhost:8002

Press Ctrl+C to stop both.
"""

import subprocess
import sys
import signal
import os

SERVICES = [
    {
        "name": "Empathy Engine",
        "cmd": [
            sys.executable, "-m", "uvicorn",
            "empathy_engine.app:app",
            "--host", "0.0.0.0",
            "--port", "8001",
            "--reload",
        ],
        "url": "http://localhost:8001",
    },
    {
        "name": "Pitch Visualizer",
        "cmd": [
            sys.executable, "-m", "uvicorn",
            "pitch_visualizer.app:app",
            "--host", "0.0.0.0",
            "--port", "8002",
            "--reload",
        ],
        "url": "http://localhost:8002",
    },
]

processes = []


def shutdown(sig, frame):
    print("\n\nShutting down services...")
    for p in processes:
        p.terminate()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("=" * 60)
    print("  darwix_ai — Starting services")
    print("=" * 60)

    for svc in SERVICES:
        p = subprocess.Popen(svc["cmd"])
        processes.append(p)
        print(f"  ✓  {svc['name']:20s}  →  {svc['url']}")

    print("=" * 60)
    print("  Press Ctrl+C to stop all services.")
    print()

    # Wait for all processes (blocks until Ctrl+C)
    for p in processes:
        p.wait()
