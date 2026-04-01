"""
TTS Engine Module for The Empathy Engine.
Synthesizes speech with emotion-driven vocal parameter modulation using pyttsx3.

FIX: Creates a fresh pyttsx3 engine per synthesis call (instead of a shared global
engine) to avoid state corruption and threading issues inside FastAPI's asyncio loop.
Synthesis is dispatched via a dedicated single-thread ThreadPoolExecutor so pyttsx3's
internal event loop never runs on the asyncio thread.
"""

import os
import uuid
import asyncio
import concurrent.futures
import pyttsx3

# Directory for audio output
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Single-worker executor: pyttsx3 is NOT thread-safe, so we serialise all
# synthesis calls through one thread.
_tts_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


# ---------------------------------------------------------------------------
# Internal synchronous helper (runs inside the executor thread)
# ---------------------------------------------------------------------------

def _run_synthesis(text: str, rate: int, volume: float, pitch: int, filepath: str) -> None:
    """Blocking pyttsx3 synthesis – must only be called from the executor thread."""
    engine = pyttsx3.init()
    try:
        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)

        # Attempt native pitch control (works on sapi5 / espeak backends)
        try:
            engine.setProperty("pitch", pitch)
        except Exception:
            pass

        # Fallback: switch voice gender based on pitch direction
        voices = engine.getProperty("voices")
        if voices:
            selected_voice = voices[0].id
            base_pitch = 100
            for voice in voices:
                name_lower = getattr(voice, "name", "").lower()
                vid_lower = getattr(voice, "id", "").lower()
                is_female = "female" in name_lower or "+f" in vid_lower or "zira" in name_lower
                is_male = (
                    "male" in name_lower and "female" not in name_lower
                ) or "+m" in vid_lower or "david" in name_lower

                if pitch > base_pitch + 10 and is_female:
                    selected_voice = voice.id
                    break
                elif pitch < base_pitch - 10 and is_male:
                    selected_voice = voice.id
                    break
            engine.setProperty("voice", selected_voice)

        engine.save_to_file(text, filepath)
        engine.runAndWait()
    finally:
        # Always stop the engine to release OS audio resources
        try:
            engine.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize_speech(text: str, rate: int = 175, volume: float = 0.8, pitch: int = 100) -> str:
    """
    Synchronous wrapper kept for backward-compatibility and CLI usage.
    In the FastAPI app use synthesize_speech_async instead.

    Returns the filename (relative to static/output/) of the generated WAV.
    """
    uid_hex = uuid.uuid4().hex
    filename = f"speech_{uid_hex[:12]}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)

    try:
        _run_synthesis(text, rate, volume, pitch, filepath)
    except Exception as e:
        # Retry once with minimal settings
        try:
            _run_synthesis(text, rate=175, volume=0.8, pitch=100, filepath=filepath)
        except Exception as e2:
            raise RuntimeError(f"TTS synthesis failed: {e2}") from e

    return filename


async def synthesize_speech_async(
    text: str, rate: int = 175, volume: float = 0.8, pitch: int = 100
) -> str:
    """
    Async-safe version: offloads blocking pyttsx3 work to the dedicated
    single-thread executor so the asyncio event loop is never blocked.

    Returns the filename (relative to static/output/) of the generated WAV.
    """
    uid_hex = uuid.uuid4().hex
    filename = f"speech_{uid_hex[:12]}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            _tts_executor,
            lambda: _run_synthesis(text, rate, volume, pitch, filepath),
        )
    except Exception as e:
        # Retry once with safe defaults
        try:
            await loop.run_in_executor(
                _tts_executor,
                lambda: _run_synthesis(text, 175, 0.8, 100, filepath),
            )
        except Exception as e2:
            raise RuntimeError(f"TTS synthesis failed: {e2}") from e

    return filename


def get_available_voices() -> list:
    """Return a list of available TTS voices (safe to call synchronously)."""
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        result = [
            {"id": v.id, "name": v.name, "languages": getattr(v, "languages", [])}
            for v in voices
        ]
        engine.stop()
        return result
    except Exception:
        return []


def cleanup_old_files(max_files: int = 50) -> None:
    """Remove oldest audio files when the output directory exceeds max_files entries."""
    try:
        files = sorted(
            [
                os.path.join(OUTPUT_DIR, f)
                for f in os.listdir(OUTPUT_DIR)
                if f.endswith(".wav")
            ],
            key=os.path.getmtime,
        )
        while len(files) > max_files:
            os.remove(files.pop(0))
    except Exception:
        pass


if __name__ == "__main__":
    print("Testing TTS engine (synchronous)...")
    fname = synthesize_speech("Hello, this is a test of the Empathy Engine!", rate=150, volume=0.9)
    print(f"Generated: {fname}")
    print(f"Full path: {os.path.join(OUTPUT_DIR, fname)}")
