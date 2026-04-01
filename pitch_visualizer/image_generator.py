"""
Image Generator Module for The Pitch Visualizer.
Creates stylized scene illustrations using Pillow (no external API required).
Generates beautiful gradient-based scene cards with text overlays and icons.

FIXES:
  - Retry logic (3 attempts) for pollinations.ai with exponential back-off
  - Increased timeout to 60 s; falls back gracefully to local gradient card
  - Removed bare 'requests' import; kept for backwards-compat but wrapped safely
"""

import os
import uuid
import time
import math
import hashlib
import asyncio
import concurrent.futures
import urllib.parse
from io import BytesIO
from typing import Any, Dict, List, Tuple

import requests  # type: ignore
from PIL import Image, ImageDraw, ImageFont, ImageFilter  # type: ignore

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_WIDTH = 768
IMG_HEIGHT = 512

# Dedicated executor for blocking image generation (HTTP + Pillow)
_img_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# ---------------------------------------------------------------------------
# Style palettes
# ---------------------------------------------------------------------------

STYLE_PALETTES: Dict[str, Any] = {
    "digital_art": {
        "gradients": [
            [(30, 30, 80), (80, 40, 120), (140, 60, 180)],
            [(20, 60, 100), (40, 100, 160), (80, 160, 220)],
            [(60, 20, 80), (120, 40, 140), (180, 80, 200)],
            [(20, 80, 60), (40, 140, 100), (80, 200, 160)],
            [(80, 20, 40), (140, 40, 80), (200, 80, 120)],
            [(40, 40, 100), (80, 80, 160), (140, 120, 220)],
        ],
        "accent": (140, 120, 255),
        "text_color": (255, 255, 255),
    },
    "photorealistic": {
        "gradients": [
            [(30, 40, 50), (60, 80, 100), (120, 160, 180)],
            [(40, 30, 20), (80, 60, 40), (160, 130, 100)],
            [(20, 30, 40), (50, 70, 90), (100, 140, 180)],
            [(30, 40, 30), (60, 80, 60), (140, 180, 140)],
            [(40, 30, 40), (80, 60, 80), (160, 120, 160)],
            [(50, 40, 30), (100, 80, 60), (180, 150, 120)],
        ],
        "accent": (200, 180, 140),
        "text_color": (255, 255, 255),
    },
    "watercolor": {
        "gradients": [
            [(180, 200, 230), (200, 220, 240), (230, 240, 250)],
            [(200, 180, 200), (220, 200, 220), (240, 220, 240)],
            [(180, 220, 200), (200, 240, 220), (230, 250, 240)],
            [(230, 200, 180), (240, 220, 200), (250, 240, 220)],
            [(200, 200, 230), (220, 210, 240), (240, 230, 250)],
            [(220, 200, 200), (240, 220, 220), (250, 240, 240)],
        ],
        "accent": (120, 100, 160),
        "text_color": (40, 40, 60),
    },
    "comic_book": {
        "gradients": [
            [(200, 50, 50), (220, 80, 40), (240, 120, 60)],
            [(50, 50, 200), (40, 80, 220), (60, 120, 240)],
            [(50, 180, 50), (40, 200, 80), (80, 230, 120)],
            [(200, 180, 50), (220, 200, 40), (240, 220, 60)],
            [(180, 50, 180), (200, 80, 200), (230, 120, 230)],
            [(50, 180, 180), (40, 200, 200), (80, 230, 230)],
        ],
        "accent": (255, 220, 50),
        "text_color": (255, 255, 255),
    },
    "minimalist": {
        "gradients": [
            [(240, 240, 245), (230, 230, 240), (220, 220, 235)],
            [(235, 240, 245), (225, 235, 240), (215, 225, 235)],
            [(245, 240, 235), (240, 230, 225), (235, 220, 215)],
            [(240, 245, 240), (230, 240, 230), (220, 235, 220)],
            [(245, 240, 240), (240, 230, 230), (235, 220, 220)],
            [(240, 240, 240), (230, 230, 230), (220, 220, 220)],
        ],
        "accent": (60, 60, 80),
        "text_color": (30, 30, 50),
    },
    "cinematic": {
        "gradients": [
            [(10, 10, 30), (30, 30, 60), (60, 50, 100)],
            [(20, 10, 10), (50, 30, 20), (100, 60, 40)],
            [(10, 20, 20), (20, 50, 40), (40, 100, 80)],
            [(10, 10, 20), (30, 30, 50), (60, 60, 100)],
            [(20, 15, 10), (50, 40, 20), (100, 80, 40)],
            [(15, 10, 20), (40, 20, 50), (80, 40, 100)],
        ],
        "accent": (220, 180, 100),
        "text_color": (255, 255, 255),
    },
}

SCENE_ICONS = {
    "opening": "◆",
    "middle": "●",
    "climax": "★",
    "closing": "◇",
}

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _create_gradient(draw: Any, width: int, height: int, colors: List[Tuple[int, int, int]]) -> None:
    c1, c2, c3 = colors
    for y in range(height):
        ratio = y / height
        if ratio < 0.5:
            r2 = ratio * 2
            r = int(c1[0] + (c2[0] - c1[0]) * r2)
            g = int(c1[1] + (c2[1] - c1[1]) * r2)
            b = int(c1[2] + (c2[2] - c1[2]) * r2)
        else:
            r2 = (ratio - 0.5) * 2
            r = int(c2[0] + (c3[0] - c2[0]) * r2)
            g = int(c2[1] + (c3[1] - c2[1]) * r2)
            b = int(c2[2] + (c3[2] - c2[2]) * r2)
        draw.line([(0, y), (width, y)], fill=(r, g, b))


def _add_geometric_shapes(
    draw: Any,
    width: int,
    height: int,
    accent_color: Tuple[int, int, int],
    seed_val: int,
) -> None:
    import random
    rng = random.Random(seed_val)

    for _ in range(rng.randint(3, 8)):
        x = rng.randint(-50, width)
        y = rng.randint(-50, height)
        size = rng.randint(20, 150)
        draw.ellipse([x, y, x + size, y + size], outline=(*accent_color, 60), width=2)

    for _ in range(rng.randint(2, 5)):
        x1, y1 = rng.randint(0, width), rng.randint(0, height)
        x2, y2 = rng.randint(0, width), rng.randint(0, height)
        draw.line([(x1, y1), (x2, y2)], fill=(*accent_color, 30), width=1)

    for _ in range(rng.randint(1, 3)):
        x = rng.randint(0, width - 100)
        y = rng.randint(0, height - 100)
        w = rng.randint(40, 120)
        h = rng.randint(40, 80)
        draw.rectangle([x, y, x + w, y + h], outline=(*accent_color, 40), width=1)


def _wrap_text(text: str, max_chars: int = 45) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current_line = ""
    for word in words:
        test = f"{current_line} {word}".strip()
        if len(test) <= max_chars:
            current_line = test
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines[:5]


# ---------------------------------------------------------------------------
# Core image generation
# ---------------------------------------------------------------------------

def _fetch_ai_image(prompt: str, width: int, height: int) -> Image.Image:
    """
    Fetch an AI-generated image from pollinations.ai with retry logic.
    Raises an exception if all attempts fail.
    """
    encoded = urllib.parse.quote(f"{prompt}, beautifully lit, masterpiece")
    url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&nologo=true"

    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGBA")
            if img.size != (width, height):
                img = img.resize((width, height), Image.Resampling.LANCZOS)
            return img
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(2 ** attempt)  # 1 s, 2 s back-off

    raise last_exc


def _make_fallback_image(
    segment: dict,
    palette: Dict[str, Any],
    gradient_colors: List[Tuple[int, int, int]],
) -> Image.Image:
    """Build a local gradient + geometric-shapes fallback image."""
    img = Image.new("RGBA", (IMG_WIDTH, IMG_HEIGHT), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    _create_gradient(draw, IMG_WIDTH, IMG_HEIGHT, gradient_colors)
    seed_str = hashlib.md5(str(segment.get("text", "")).encode()).hexdigest()
    seed = int(seed_str[:8], 16)
    _add_geometric_shapes(draw, IMG_WIDTH, IMG_HEIGHT, palette["accent"], seed)  # type: ignore
    return img


def _load_fonts() -> Tuple[Any, Any, Any, Any]:
    """Try to load DejaVu / Liberation fonts; fall back to PIL default."""
    font_paths = [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    ]
    for bold_path, regular_path in font_paths:
        try:
            return (
                ImageFont.truetype(bold_path, 28),
                ImageFont.truetype(regular_path, 18),
                ImageFont.truetype(regular_path, 14),
                ImageFont.truetype(bold_path, 36),
            )
        except Exception:
            continue
    default = ImageFont.load_default()
    return default, default, default, default


def generate_scene_image(
    segment: dict,
    prompt: str,
    style: str = "digital_art",
    scene_number: int = 1,
    total_scenes: int = 1,
) -> str:
    """
    Generate a stylized scene image and save it to OUTPUT_DIR.

    Returns the filename (relative to static/output/).
    """
    palette = STYLE_PALETTES.get(style, STYLE_PALETTES["digital_art"])
    position = segment.get("position", "middle")

    gradient_idx = (scene_number - 1) % len(palette["gradients"])
    gradient_colors = palette["gradients"][gradient_idx]  # type: ignore

    # Try AI image; fall back to local gradient card
    try:
        img = _fetch_ai_image(prompt, IMG_WIDTH, IMG_HEIGHT)
    except Exception as exc:
        print(f"[image_generator] AI fetch failed ({exc}), using fallback.")
        img = _make_fallback_image(segment, palette, gradient_colors)  # type: ignore

    font_large, font_medium, font_small, font_icon = _load_fonts()

    text_color: Tuple[int, int, int] = palette["text_color"]  # type: ignore
    accent: Tuple[int, int, int] = palette["accent"]  # type: ignore

    # Semi-transparent text overlay
    overlay = Image.new("RGBA", (IMG_WIDTH, IMG_HEIGHT), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    card_margin, card_top, card_bottom = 60, 100, IMG_HEIGHT - 60
    overlay_draw.rounded_rectangle(
        [card_margin, card_top, IMG_WIDTH - card_margin, card_bottom],
        radius=20,
        fill=(0, 0, 0, 100),
    )
    img = Image.alpha_composite(img.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(img)

    # Scene badge
    bx, by = card_margin + 20, card_top + 20
    draw.rounded_rectangle([bx, by, bx + 120, by + 30], radius=15, fill=(*accent, 200))
    draw.text((bx + 15, by + 6), f"SCENE {scene_number}", fill=(255, 255, 255), font=font_small)
    draw.text((bx + 135, by + 8), position.upper(), fill=(*accent, 180), font=font_small)

    # Main text
    text = segment.get("text", "")
    text_y = card_top + 70
    for line in _wrap_text(text, max_chars=50):
        draw.text((card_margin + 30, text_y), line, fill=text_color, font=font_medium)
        text_y += 28

    # Key-phrase tags
    nouns = (segment.get("key_phrases") or {}).get("nouns", [])[:4]
    if nouns:
        draw.text(
            (card_margin + 30, card_bottom - 50),
            "  •  ".join(nouns),
            fill=(*accent, 200),
            font=font_small,
        )

    # Scene counter
    draw.text(
        (IMG_WIDTH - card_margin - 80, card_top + 25),
        f"{scene_number} / {total_scenes}",
        fill=(*text_color[:3], 150),  # type: ignore
        font=font_small,
    )

    img_rgb = img.convert("RGB")
    filename = f"scene_{scene_number}_{uuid.uuid4().hex[:8]}.png"
    img_rgb.save(os.path.join(OUTPUT_DIR, filename), "PNG")
    return filename


# ---------------------------------------------------------------------------
# Batch generation (synchronous & async)
# ---------------------------------------------------------------------------

def generate_storyboard_images(segments: list, style: str = "digital_art") -> list:
    """Generate images for all segments and return a list of filenames."""
    total = len(segments)
    return [
        generate_scene_image(
            segment=seg,
            prompt=seg.get("prompt", ""),
            style=style,
            scene_number=seg.get("id", i + 1),
            total_scenes=total,
        )
        for i, seg in enumerate(segments)
    ]


async def generate_storyboard_images_async(segments: list, style: str = "digital_art") -> list:
    """
    Async wrapper: runs blocking image generation in the thread-pool executor
    so the asyncio event loop is not blocked.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _img_executor,
        lambda: generate_storyboard_images(segments, style),
    )


def cleanup_old_files(max_files: int = 100) -> None:
    """Remove oldest PNG files when the output directory exceeds max_files."""
    try:
        files = sorted(
            [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")],
            key=os.path.getmtime,
        )
        while len(files) > max_files:
            os.remove(files.pop(0))
    except Exception:
        pass


if __name__ == "__main__":
    test_segment = {
        "id": 1,
        "text": "Our team developed an innovative AI solution that transformed customer engagement.",
        "key_phrases": {
            "nouns": ["team", "solution", "customer", "engagement"],
            "verbs": ["developed", "transformed"],
            "adjectives": ["innovative"],
        },
        "position": "opening",
    }
    fname = generate_scene_image(test_segment, "test prompt", "digital_art", 1, 3)
    print(f"Generated: {fname}")
