"""
The Empathy Engine - FastAPI Application
Emotion-driven Text-to-Speech service that dynamically modulates
vocal characteristics based on detected emotion.

FIX: TTS synthesis now uses synthesize_speech_async which dispatches the
blocking pyttsx3 call to a dedicated ThreadPoolExecutor, keeping the
asyncio event loop fully responsive.
"""

import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from empathy_engine.emotion_detector import detect_emotion
from empathy_engine.voice_modulator import get_voice_parameters, generate_ssml
from empathy_engine.tts_engine import synthesize_speech_async, cleanup_old_files

app = FastAPI(
    title="The Empathy Engine",
    description="AI-driven Text-to-Speech that gives AI a human voice through emotion detection",
    version="1.0.0",
)

# Setup templates and static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the web UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze_text(request: Request):
    """
    Analyze text for emotion and generate modulated speech.

    Accepts JSON body: {"text": "your text here"}
    Returns emotion analysis, voice parameters, SSML, and audio file URL.
    """
    try:
        body = await request.json()
        text = body.get("text", "").strip()

        if not text:
            return JSONResponse({"error": "No text provided"}, status_code=400)

        if len(text) > 5000:
            return JSONResponse(
                {"error": "Text too long (max 5000 characters)"}, status_code=400
            )

        # Step 1: Detect emotion
        emotion_result = detect_emotion(text)

        # Step 2: Get voice parameters
        voice_params = get_voice_parameters(
            emotion_result["emotion"],
            emotion_result["intensity"],
        )

        # Step 3: Generate SSML (returned in response for cloud TTS integration;
        # pyttsx3 uses the raw modulated parameters directly)
        ssml = generate_ssml(text, emotion_result["emotion"], emotion_result["intensity"])

        # Step 4: Synthesize speech (async-safe – does NOT block the event loop)
        cleanup_old_files()
        audio_filename = await synthesize_speech_async(
            text,
            rate=voice_params["rate"],
            volume=voice_params["volume"],
            pitch=voice_params["pitch"],
        )

        audio_url = f"/static/output/{audio_filename}"

        return JSONResponse(
            {
                "success": True,
                "text": text,
                "emotion": {
                    "detected": emotion_result["emotion"],
                    "category": emotion_result["category"],
                    "intensity": emotion_result["intensity"],
                    "vader_scores": emotion_result["vader_scores"],
                    "textblob_polarity": emotion_result["textblob_polarity"],
                    "textblob_subjectivity": emotion_result["textblob_subjectivity"],
                },
                "voice_parameters": {
                    "rate": voice_params["rate"],
                    "pitch": voice_params["pitch"],
                    "volume": voice_params["volume"],
                    "description": voice_params["description"],
                    "base_rate": voice_params["base_rate"],
                    "base_pitch": voice_params["base_pitch"],
                    "base_volume": voice_params["base_volume"],
                },
                "ssml": ssml,
                "audio_url": audio_url,
            }
        )

    except Exception as e:
        return JSONResponse(
            {"error": f"Processing failed: {str(e)}"}, status_code=500
        )


@app.get("/voices")
async def list_voices():
    """Return available TTS voices on this machine."""
    from empathy_engine.tts_engine import get_available_voices
    return {"voices": get_available_voices()}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "empathy-engine"}
