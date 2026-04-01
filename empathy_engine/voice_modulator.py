"""
Voice Modulation Module for The Empathy Engine.
Maps detected emotions to vocal parameters (rate, pitch, volume) with intensity scaling.
"""

# Base voice parameters
BASE_RATE = 175  # words per minute
BASE_PITCH = 100  # Hz modifier (relative)
BASE_VOLUME = 0.8  # 0.0-1.0

# Emotion-to-voice parameter mapping
# Each emotion defines deltas from base values at maximum intensity
# Actual deltas are scaled by the detected intensity
EMOTION_VOICE_MAP = {
    "excited": {
        "rate_delta": 50,       # Speak faster when excited
        "pitch_delta": 40,      # Higher pitch
        "volume_delta": 0.2,    # Louder
        "description": "Fast, high-pitched, energetic delivery"
    },
    "happy": {
        "rate_delta": 25,       # Slightly faster
        "pitch_delta": 20,      # Slightly higher pitch
        "volume_delta": 0.1,    # Slightly louder
        "description": "Warm, upbeat, pleasant delivery"
    },
    "neutral": {
        "rate_delta": 0,        # Normal speed
        "pitch_delta": 0,       # Normal pitch
        "volume_delta": 0.0,    # Normal volume
        "description": "Calm, steady, professional delivery"
    },
    "concerned": {
        "rate_delta": -15,      # Slightly slower
        "pitch_delta": -10,     # Slightly lower
        "volume_delta": -0.05,  # Slightly softer
        "description": "Careful, measured, empathetic delivery"
    },
    "frustrated": {
        "rate_delta": 30,       # Faster, agitated
        "pitch_delta": -20,     # Lower, tense pitch
        "volume_delta": 0.15,   # Louder
        "description": "Tense, clipped, forceful delivery"
    },
    "sad": {
        "rate_delta": -40,      # Much slower
        "pitch_delta": -30,     # Much lower pitch
        "volume_delta": -0.15,  # Softer
        "description": "Slow, low, gentle delivery"
    },
    "surprised": {
        "rate_delta": 20,       # Slightly faster
        "pitch_delta": 35,      # Higher pitch
        "volume_delta": 0.1,    # Slightly louder
        "description": "Quick, high-pitched, breathless delivery"
    },
    "inquisitive": {
        "rate_delta": -10,      # Slightly slower
        "pitch_delta": 15,      # Slightly higher (questioning)
        "volume_delta": 0.0,    # Normal volume
        "description": "Thoughtful, rising-tone delivery"
    },
}


def get_voice_parameters(emotion: str, intensity: float) -> dict:
    """
    Calculate voice parameters based on detected emotion and intensity.

    Args:
        emotion: Detected emotion string
        intensity: Emotion intensity (0.0-1.0)

    Returns:
        dict with:
            - rate: int (words per minute)
            - pitch: int (Hz modifier)
            - volume: float (0.0-1.0)
            - description: str (human-readable description)
            - emotion: str
            - intensity: float
    """
    emotion_config = EMOTION_VOICE_MAP.get(emotion, EMOTION_VOICE_MAP["neutral"])

    # Apply intensity scaling
    scaled_rate_delta = float(emotion_config["rate_delta"]) * intensity  # type: ignore
    scaled_pitch_delta = float(emotion_config["pitch_delta"]) * intensity  # type: ignore
    scaled_volume_delta = float(emotion_config["volume_delta"]) * intensity  # type: ignore

    # Calculate final parameters
    rate = max(80, min(300, int(BASE_RATE + scaled_rate_delta)))
    pitch = max(40, min(200, int(BASE_PITCH + scaled_pitch_delta)))
    volume = max(0.2, min(1.0, round(BASE_VOLUME + scaled_volume_delta, 2)))  # type: ignore

    return {
        "rate": rate,
        "pitch": pitch,
        "volume": volume,
        "description": emotion_config["description"],
        "emotion": emotion,
        "intensity": intensity,
        "base_rate": BASE_RATE,
        "base_pitch": BASE_PITCH,
        "base_volume": BASE_VOLUME,
    }


def generate_ssml(text: str, emotion: str, intensity: float) -> str:
    """
    Generate SSML markup based on emotion and intensity.

    Args:
        text: The text to markup
        emotion: Detected emotion
        intensity: Emotion intensity (0.0-1.0)

    Returns:
        SSML-formatted string
    """
    params = get_voice_parameters(emotion, intensity)

    # Map rate to SSML prosody rate attribute
    if params["rate"] > BASE_RATE + 40:
        rate_attr = "x-fast"
    elif params["rate"] > BASE_RATE + 20:
        rate_attr = "fast"
    elif params["rate"] < BASE_RATE - 40:
        rate_attr = "x-slow"
    elif params["rate"] < BASE_RATE - 20:
        rate_attr = "slow"
    else:
        rate_attr = "medium"

    # Map pitch to SSML
    if params["pitch"] > BASE_PITCH + 30:
        pitch_attr = "x-high"
    elif params["pitch"] > BASE_PITCH + 15:
        pitch_attr = "high"
    elif params["pitch"] < BASE_PITCH - 30:
        pitch_attr = "x-low"
    elif params["pitch"] < BASE_PITCH - 15:
        pitch_attr = "low"
    else:
        pitch_attr = "medium"

    # Map volume
    if params["volume"] > 0.85:
        volume_attr = "loud"
    elif params["volume"] < 0.6:
        volume_attr = "soft"
    else:
        volume_attr = "medium"

    # Add emphasis for high-intensity emotions
    emphasis = ""
    if intensity > 0.7:
        emphasis = ' level="strong"'
    elif intensity > 0.4:
        emphasis = ' level="moderate"'

    ssml = f"""<speak>
  <prosody rate="{rate_attr}" pitch="{pitch_attr}" volume="{volume_attr}">
    <emphasis{emphasis}>{text}</emphasis>
  </prosody>
</speak>"""

    return ssml


if __name__ == "__main__":
    # Test voice parameters
    test_cases = [
        ("excited", 0.9),
        ("happy", 0.6),
        ("neutral", 0.3),
        ("frustrated", 0.8),
        ("sad", 0.7),
        ("surprised", 0.5),
    ]

    for emotion, intensity in test_cases:
        params = get_voice_parameters(emotion, intensity)
        print(f"\n{emotion} (intensity={intensity}):")
        print(f"  Rate: {params['rate']} wpm")
        print(f"  Pitch: {params['pitch']} Hz")
        print(f"  Volume: {params['volume']}")
        print(f"  Description: {params['description']}")

        ssml = generate_ssml("Hello, how are you?", emotion, intensity)
        print(f"  SSML: {ssml}")
