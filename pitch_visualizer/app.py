"""
The Pitch Visualizer - FastAPI Application
Narrative-to-storyboard service that ingests text, segments it into scenes,
engineers image prompts, and generates a visual storyboard.

FIX: image generation now uses generate_storyboard_images_async, which
offloads the blocking HTTP + Pillow work to a ThreadPoolExecutor so the
asyncio event loop is never stalled.
"""

import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pitch_visualizer.segmenter import segment_text
from pitch_visualizer.prompt_engineer import generate_prompts_for_storyboard, get_available_styles
from pitch_visualizer.image_generator import (
    generate_storyboard_images_async,
    cleanup_old_files,
)

app = FastAPI(
    title="The Pitch Visualizer",
    description="Transform narrative text into visual storyboards with AI-powered prompt engineering",
    version="1.0.0",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the web UI."""
    styles = get_available_styles()
    return templates.TemplateResponse("index.html", {"request": request, "styles": styles})


@app.get("/styles")
async def list_styles():
    """Return available visual styles."""
    return {"styles": get_available_styles()}


@app.post("/generate")
async def generate_storyboard(request: Request):
    """
    Generate a visual storyboard from narrative text.

    Accepts JSON body: {"text": "...", "style": "digital_art"}
    Returns storyboard panels with images and prompts.
    """
    try:
        body = await request.json()
        text = body.get("text", "").strip()
        style = body.get("style", "digital_art")

        if not text:
            return JSONResponse({"error": "No text provided"}, status_code=400)

        if len(text) > 10000:
            return JSONResponse(
                {"error": "Text too long (max 10000 characters)"}, status_code=400
            )

        # Step 1: Segment the narrative text
        segments = segment_text(text)
        if not segments:
            return JSONResponse(
                {
                    "error": (
                        "Could not segment the text. "
                        "Please provide narrative text with multiple sentences."
                    )
                },
                status_code=400,
            )

        # Step 2: Generate enhanced prompts
        enriched_segments = generate_prompts_for_storyboard(segments, style)

        # Step 3: Generate images (async-safe – does NOT block the event loop)
        cleanup_old_files()
        image_filenames = await generate_storyboard_images_async(enriched_segments, style)

        # Step 4: Build response
        panels = [
            {
                "scene_number": seg["id"],
                "text": seg["text"],
                "position": seg["position"],
                "key_phrases": seg["key_phrases"],
                "prompt": seg["prompt"],
                "style": seg["style_name"],
                "image_url": f"/static/output/{image_filenames[i]}",
            }
            for i, seg in enumerate(enriched_segments)
        ]

        return JSONResponse(
            {
                "success": True,
                "total_panels": len(panels),
                "style": style,
                "panels": panels,
            }
        )

    except Exception as e:
        return JSONResponse(
            {"error": f"Generation failed: {str(e)}"}, status_code=500
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "pitch-visualizer"}
