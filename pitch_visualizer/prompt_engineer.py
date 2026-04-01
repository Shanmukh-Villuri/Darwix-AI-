"""
Intelligent Prompt Engineering Module for The Pitch Visualizer.
Transforms narrative text segments into visually descriptive image generation prompts.

FIX: Updated Gemini SDK usage to be compatible with google-generativeai >= 0.7.
     The import guard and try/except fallback are retained so the module works
     even without a GEMINI_API_KEY set.
"""

import os

# Optional Gemini support for LLM-powered prompt refinement
try:
    import google.generativeai as genai  # type: ignore
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    genai = None

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

VISUAL_STYLES = {
    "digital_art": {
        "name": "Digital Art",
        "suffix": "digital art style, vibrant colors, modern illustration, clean lines, high quality",
        "color_palette": "vibrant and modern",
    },
    "photorealistic": {
        "name": "Photorealistic",
        "suffix": "photorealistic, professional photography, cinematic lighting, ultra detailed, 8k",
        "color_palette": "natural and realistic",
    },
    "watercolor": {
        "name": "Watercolor",
        "suffix": "watercolor painting style, soft edges, flowing colors, artistic, hand-painted look",
        "color_palette": "soft and flowing",
    },
    "comic_book": {
        "name": "Comic Book",
        "suffix": "comic book style, bold outlines, dynamic composition, halftone dots, action panels",
        "color_palette": "bold and dramatic",
    },
    "minimalist": {
        "name": "Minimalist",
        "suffix": "minimalist design, clean composition, limited color palette, modern, elegant simplicity",
        "color_palette": "clean and minimal",
    },
    "cinematic": {
        "name": "Cinematic",
        "suffix": "cinematic scene, dramatic lighting, movie still, wide angle, atmospheric, film grain",
        "color_palette": "dramatic and moody",
    },
}

POSITION_DESCRIPTORS = {
    "opening": {
        "mood": "establishing, introductory",
        "composition": "wide establishing shot",
        "lighting": "warm morning light",
    },
    "middle": {
        "mood": "developing, building tension",
        "composition": "medium shot, focused",
        "lighting": "natural daylight",
    },
    "climax": {
        "mood": "dramatic, intense, pivotal",
        "composition": "close-up, dynamic angle",
        "lighting": "dramatic spotlight",
    },
    "closing": {
        "mood": "resolving, triumphant, reflective",
        "composition": "wide panoramic shot",
        "lighting": "golden hour, warm glow",
    },
}

CONCEPT_VISUALS = {
    "team": "group of diverse professionals working together",
    "client": "business meeting with stakeholders",
    "growth": "upward trending graph, flourishing growth",
    "success": "celebration, achievement, victory",
    "innovation": "futuristic technology, glowing circuits",
    "struggle": "person climbing steep mountain",
    "challenge": "obstacle course, complex maze",
    "solution": "lightbulb moment, puzzle pieces fitting together",
    "technology": "sleek digital interface, holographic display",
    "customer": "happy customer interacting with product",
    "data": "flowing streams of data, digital visualization",
    "strategy": "chess pieces, strategic map",
    "transform": "butterfly emerging, metamorphosis",
    "revenue": "rising financial charts, golden coins",
    "product": "elegant product showcase, spotlight display",
    "market": "bustling marketplace, global connections",
    "future": "futuristic cityscape, horizon at dawn",
    "partnership": "handshake, two puzzle pieces connecting",
    "leadership": "person at summit, guiding light",
    "impact": "ripple effect in water, shockwave",
}

# ---------------------------------------------------------------------------
# Gemini helper
# ---------------------------------------------------------------------------

def _gemini_refine(segment: dict, style_name: str, position_config: dict) -> str | None:
    """
    Use Google Gemini to produce a highly detailed image prompt.
    Returns None if Gemini is unavailable or the call fails.

    Compatible with google-generativeai >= 0.5 (0.x SDK).
    For the newer google-genai 1.x SDK, update the import accordingly.
    """
    if not HAS_GEMINI or genai is None:
        return None

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None

    try:
        genai.configure(api_key=api_key)  # type: ignore

        # Try newer model names first, fall back to older ones
        for model_name in ("gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-pro"):
            try:
                model = genai.GenerativeModel(model_name)  # type: ignore
                system_instruction = (
                    f"You are an expert AI image prompt engineer. Transform the narrative sentence "
                    f"below into a highly detailed, artistic prompt for a text-to-image model "
                    f"(Midjourney / DALL-E / Stable Diffusion).\n"
                    f"Style: {style_name}. "
                    f"Composition: {position_config['composition']}. "
                    f"Lighting: {position_config['lighting']}. "
                    f"Mood: {position_config['mood']}.\n"
                    f"Output ONLY the prompt as comma-separated descriptors. No preamble, no quotes."
                )
                response = model.generate_content(
                    f"{system_instruction}\n\nSentence: \"{segment.get('text', '')}\""
                )
                result = response.text.strip().replace('"', "").replace("\n", " ")
                if len(result) > 20:
                    return result
            except Exception:
                continue

    except Exception as exc:
        print(f"[prompt_engineer] Gemini call failed: {exc}")

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_prompt(segment: dict, style: str = "digital_art") -> str:
    """
    Generate a visually descriptive image prompt from a text segment.

    Tries Gemini LLM refinement first; falls back to NLP-based construction.
    """
    style_config = VISUAL_STYLES.get(style, VISUAL_STYLES["digital_art"])
    position_key = str(segment.get("position", "middle"))
    position_config = POSITION_DESCRIPTORS.get(position_key, POSITION_DESCRIPTORS["middle"])

    # --- Try LLM refinement ---
    llm_prompt = _gemini_refine(segment, style_config["name"], position_config)
    if llm_prompt:
        return f"{llm_prompt}, {style_config['suffix']}, consistent style, high quality"

    # --- NLP-based fallback ---
    nouns = segment.get("key_phrases", {}).get("nouns", [])
    adjectives = segment.get("key_phrases", {}).get("adjectives", [])

    visual_elements: list[str] = []
    for noun in nouns:
        noun_str = str(noun)
        for concept, visual in CONCEPT_VISUALS.items():
            if concept in noun_str or noun_str in concept:
                visual_elements.append(visual)
                break
        else:
            visual_elements.append(noun_str)

    text = str(segment.get("text", ""))
    if len(text) > 100:
        scene_desc = f"A scene depicting: {', '.join(visual_elements[:3])}"
    else:
        scene_desc = f"A scene showing {text.lower().rstrip('.')}"

    parts = [scene_desc]
    if adjectives:
        parts.append(f"atmosphere: {', '.join(adjectives)}")
    parts.append(f"{position_config['composition']}, {position_config['lighting']}")
    parts.append(f"mood: {position_config['mood']}")
    parts.append(style_config["suffix"])
    parts.append("consistent style throughout, professional quality")

    return ", ".join(parts)


def generate_prompts_for_storyboard(segments: list, style: str = "digital_art") -> list:
    """Generate enriched prompts for every segment."""
    return [
        {
            **seg,
            "prompt": generate_prompt(seg, style),
            "style": style,
            "style_name": VISUAL_STYLES.get(style, VISUAL_STYLES["digital_art"])["name"],
        }
        for seg in segments
    ]


def get_available_styles() -> list:
    """Return list of available visual styles."""
    return [
        {"id": key, "name": val["name"], "description": val["suffix"]}
        for key, val in VISUAL_STYLES.items()
    ]


if __name__ == "__main__":
    test_segment = {
        "id": 1,
        "text": "Our team developed an innovative AI solution that transformed customer engagement.",
        "key_phrases": {
            "nouns": ["team", "solution", "customer", "engagement"],
            "verbs": ["developed", "transformed"],
            "adjectives": ["innovative"],
        },
        "position": "middle",
    }
    for style_key in VISUAL_STYLES:
        print(f"\n[{style_key}]")
        print(generate_prompt(test_segment, style_key))
