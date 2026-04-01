# darwix_ai — The Empathy Engine & The Pitch Visualizer

Two AI-powered services built for the Darwix AI Challenge.

---

## 🎙️ Challenge 1: The Empathy Engine

**Giving AI a Human Voice** — A FastAPI service that dynamically modulates synthesized speech based on the detected emotion of source text.

### Features
- **Emotion Detection** — VADER + TextBlob classify text into 8 granular emotions: `happy`, `excited`, `neutral`, `concerned`, `frustrated`, `sad`, `surprised`, `inquisitive`
- **Intensity Scaling** — Detected intensity (0.0–1.0) scales every modulation delta proportionally
- **Voice Parameter Modulation** — Adjusts `rate` (wpm), `pitch` (Hz modifier)\*, and `volume` via pyttsx3
- **Audio Output** — Generates `.wav` files saved under `empathy_engine/static/output/`
- **SSML** — Full SSML payload returned in the API response (ready for AWS Polly / GCP TTS)
- **Web UI** — Dark-mode glassmorphism interface with embedded audio player

> \*Native pitch control depends on the OS backend (espeak-ng fully supports it; macOS/Windows SAPI5 may vary). The engine compensates via voice-gender switching.

### Emotion → Voice Mapping

| Emotion     | Rate Δ   | Pitch Δ  | Volume Δ | Style                         |
|-------------|----------|----------|----------|-------------------------------|
| Excited     | +50 wpm  | +40 Hz   | +0.20    | Fast, high-pitched, energetic |
| Happy       | +25 wpm  | +20 Hz   | +0.10    | Warm, upbeat, pleasant        |
| Neutral     | ±0       | ±0       | ±0.00    | Calm, steady, professional    |
| Concerned   | −15 wpm  | −10 Hz   | −0.05    | Careful, measured, empathetic |
| Frustrated  | +30 wpm  | −20 Hz   | +0.15    | Tense, clipped, forceful      |
| Sad         | −40 wpm  | −30 Hz   | −0.15    | Slow, low, gentle             |
| Surprised   | +20 wpm  | +35 Hz   | +0.10    | Quick, high-pitched, breathless|
| Inquisitive | −10 wpm  | +15 Hz   | ±0.00    | Thoughtful, rising-tone       |

*Deltas are at maximum intensity and are scaled by the detected intensity value.*

---

## 🖼️ Challenge 2: The Pitch Visualizer

**From Words to Storyboard** — A FastAPI service that ingests narrative text, segments it into key scenes, engineers image prompts, and generates a multi-panel visual storyboard.

### Features
- **Narrative Segmentation** — NLTK sentence tokenization + topic-shift detection groups text into 3–6 logical scenes
- **Intelligent Prompt Engineering** — POS tagging + concept-to-visual mapping builds rich prompts; Gemini LLM refinement when `GEMINI_API_KEY` is set
- **AI Image Generation** — Calls pollinations.ai (free, keyless) with 3-attempt retry + exponential back-off; local gradient-card fallback if unavailable
- **6 Visual Styles** — Digital Art, Photorealistic, Watercolor, Comic Book, Minimalist, Cinematic
- **Dynamic Web UI** — Dark-mode interface with animated storyboard panels and per-panel prompt preview

### Prompt Engineering Pipeline
1. Extract key nouns, verbs, adjectives via NLTK POS tagging
2. Map concepts to visual descriptions (`"team"` → `"group of diverse professionals"`)
3. Apply position-based composition descriptors (opening = wide shot, climax = close-up)
4. Append consistent style suffix for visual cohesion
5. (Optional) LLM supercharge via Gemini when `GEMINI_API_KEY` is set

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- espeak-ng (Linux TTS backend)

```bash
# Ubuntu / Debian
sudo apt-get install espeak-ng

# macOS (uses built-in 'say' via pyttsx3 – no extra install needed)
# Windows (uses SAPI5 via pyttsx3 – no extra install needed)
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### Download NLTK data (auto-downloaded on first run, or run manually)

```bash
python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
```

---

## ▶️ Running the Services

Run both services from the **project root** (the folder containing `empathy_engine/` and `pitch_visualizer/`):

**Option A — convenience script (starts both):**

```bash
python3 run.py
```

**Option B — run individually:**

```bash
# The Empathy Engine  →  http://localhost:8001
python3 -m uvicorn empathy_engine.app:app --host 0.0.0.0 --port 8001 --reload

# The Pitch Visualizer  →  http://localhost:8002
python3 -m uvicorn pitch_visualizer.app:app --host 0.0.0.0 --port 8002 --reload
```

---

## 🔑 Optional: Gemini API Key

To enable LLM-powered prompt refinement in the Pitch Visualizer:

```bash
export GEMINI_API_KEY="your_key_here"
```

Without the key the service falls back to its NLP-based prompt builder automatically.

---

## 📂 Project Structure

```
darwix_ai/
├── empathy_engine/              # Challenge 1
│   ├── __init__.py
│   ├── app.py                   # FastAPI application
│   ├── emotion_detector.py      # VADER + TextBlob emotion analysis
│   ├── voice_modulator.py       # Emotion → voice parameters + SSML
│   ├── tts_engine.py            # pyttsx3 speech synthesis (async-safe)
│   ├── templates/index.html     # Web UI
│   └── static/output/           # Generated audio files
├── pitch_visualizer/            # Challenge 2
│   ├── __init__.py
│   ├── app.py                   # FastAPI application
│   ├── segmenter.py             # NLTK narrative segmentation
│   ├── prompt_engineer.py       # Intelligent prompt generation (+ Gemini)
│   ├── image_generator.py       # Pillow scene cards + pollinations.ai
│   ├── templates/index.html     # Web UI
│   └── static/output/           # Generated storyboard images
├── run.py                       # Convenience launcher (both services)
├── requirements.txt
└── README.md
```

---

## 🧪 API Reference

### Empathy Engine

**POST /analyze**
```json
{ "text": "I am so excited about this amazing opportunity!" }
```

Response:
```json
{
  "success": true,
  "emotion": { "detected": "excited", "category": "positive", "intensity": 0.92 },
  "voice_parameters": { "rate": 221, "pitch": 137, "volume": 0.98 },
  "ssml": "<speak>...</speak>",
  "audio_url": "/static/output/speech_abc123.wav"
}
```

**GET /voices** — list available TTS voices on this machine

### Pitch Visualizer

**POST /generate**
```json
{ "text": "Our client was struggling. We innovated. They succeeded.", "style": "digital_art" }
```

Response:
```json
{
  "success": true,
  "total_panels": 3,
  "panels": [
    {
      "scene_number": 1,
      "text": "Our client was struggling.",
      "prompt": "A scene showing...",
      "image_url": "/static/output/scene_1_abc123.png"
    }
  ]
}
```

**GET /styles** — list available visual styles

---

## 🛠️ Tech Stack

| Component           | Technology                       |
|---------------------|----------------------------------|
| Backend             | Python 3.10+, FastAPI            |
| Emotion Analysis    | VADER, TextBlob                  |
| TTS                 | pyttsx3 + espeak-ng              |
| Text Segmentation   | NLTK                             |
| Image Generation    | pollinations.ai + Pillow (PIL)   |
| LLM Prompt Boost    | Google Gemini (optional)         |
| Frontend            | HTML5, CSS3 (glassmorphism), JS  |
| Templating          | Jinja2                           |
