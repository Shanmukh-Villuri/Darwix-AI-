"""
Emotion Detection Module for The Empathy Engine.
Uses VADER and TextBlob for granular emotion classification with intensity scaling.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Initialize VADER analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Emotion keywords for granular classification
EMOTION_KEYWORDS = {
    "excited": [
        "amazing", "fantastic", "incredible", "awesome", "wonderful", "thrilled",
        "ecstatic", "best", "love", "excellent", "brilliant", "outstanding",
        "extraordinary", "magnificent", "superb", "perfect", "greatest"
    ],
    "happy": [
        "happy", "glad", "pleased", "good", "nice", "great", "enjoy",
        "delighted", "cheerful", "joyful", "content", "satisfied", "thankful",
        "grateful", "blessed", "fortunate", "wonderful"
    ],
    "concerned": [
        "worried", "concerned", "anxious", "uncertain", "unsure", "nervous",
        "uneasy", "doubtful", "hesitant", "troubled", "apprehensive", "wary",
        "cautious", "skeptical", "puzzled"
    ],
    "frustrated": [
        "frustrated", "angry", "annoyed", "irritated", "furious", "mad",
        "outraged", "disgusted", "hate", "terrible", "awful", "horrible",
        "worst", "unacceptable", "ridiculous", "pathetic"
    ],
    "sad": [
        "sad", "unhappy", "depressed", "disappointed", "heartbroken", "miserable",
        "sorrowful", "gloomy", "devastated", "crushed", "hopeless", "lonely",
        "lost", "crying", "tears", "grief", "mourning"
    ],
    "surprised": [
        "surprised", "shocked", "astonished", "amazed", "stunned", "unexpected",
        "unbelievable", "wow", "whoa", "really", "seriously", "can't believe"
    ],
    "inquisitive": [
        "why", "how", "what", "wondering", "curious", "question", "confused",
        "understand", "explain", "tell me", "help me", "figure out"
    ]
}


def _count_keyword_matches(text_lower: str) -> dict:
    """Count keyword matches for each emotion category."""
    scores = {}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        scores[emotion] = count
    return scores


def _classify_granular_emotion(vader_scores: dict, text: str) -> tuple:
    """
    Classify text into granular emotion categories.
    Returns (emotion_name, intensity, category).
    """
    compound = vader_scores["compound"]
    pos = vader_scores["pos"]
    neg = vader_scores["neg"]
    neu = vader_scores["neu"]

    text_lower = text.lower()
    keyword_scores = _count_keyword_matches(text_lower)

    # Determine primary emotion based on compound score + keywords
    if compound >= 0.6:
        # Strong positive
        if keyword_scores.get("excited", 0) > keyword_scores.get("happy", 0):
            emotion = "excited"
            category = "positive"
        else:
            emotion = "happy"
            category = "positive"
        intensity = min(1.0, (compound + 1) / 2 * 1.2)

    elif compound >= 0.2:
        # Mild positive
        emotion = "happy"
        category = "positive"
        intensity = (compound + 1) / 2

    elif compound <= -0.6:
        # Strong negative
        if keyword_scores.get("frustrated", 0) > keyword_scores.get("sad", 0):
            emotion = "frustrated"
            category = "negative"
        else:
            emotion = "sad"
            category = "negative"
        intensity = min(1.0, abs(compound) * 1.2)

    elif compound <= -0.2:
        # Mild negative
        if keyword_scores.get("concerned", 0) > 0:
            emotion = "concerned"
            category = "negative"
        elif keyword_scores.get("frustrated", 0) > keyword_scores.get("sad", 0):
            emotion = "frustrated"
            category = "negative"
        else:
            emotion = "sad"
            category = "negative"
        intensity = abs(compound)

    else:
        # Neutral zone - check for special emotions
        if keyword_scores.get("surprised", 0) > 0:
            emotion = "surprised"
            category = "neutral"
            intensity = 0.5 + keyword_scores["surprised"] * 0.1
        elif keyword_scores.get("inquisitive", 0) > 0:
            emotion = "inquisitive"
            category = "neutral"
            intensity = 0.4 + keyword_scores["inquisitive"] * 0.1
        elif keyword_scores.get("concerned", 0) > 0:
            emotion = "concerned"
            category = "negative"
            intensity = 0.4
        else:
            emotion = "neutral"
            category = "neutral"
            intensity = 0.3

    # Override with keyword-based detection if strong signal
    if keyword_scores:
        max_kw_emotion = max(keyword_scores.items(), key=lambda x: x[1])[0]
    else:
        max_kw_emotion = "neutral"
    if keyword_scores[max_kw_emotion] >= 3:
        emotion = max_kw_emotion
        if emotion in ("excited", "happy"):
            category = "positive"
        elif emotion in ("frustrated", "sad", "concerned"):
            category = "negative"
        else:
            category = "neutral"
        intensity = min(1.0, 0.5 + keyword_scores[max_kw_emotion] * 0.1)

    intensity = max(0.1, min(1.0, intensity))

    return emotion, round(intensity, 2), category


def detect_emotion(text: str) -> dict:
    """
    Analyze text and detect emotion with granular classification.

    Returns:
        dict with keys:
            - emotion: str (happy, excited, neutral, concerned, frustrated, sad, surprised, inquisitive)
            - intensity: float (0.0-1.0)
            - category: str (positive, negative, neutral)
            - vader_scores: dict (compound, pos, neg, neu)
            - textblob_polarity: float
            - textblob_subjectivity: float
    """
    if not text or not text.strip():
        return {
            "emotion": "neutral",
            "intensity": 0.0,
            "category": "neutral",
            "vader_scores": {"compound": 0, "pos": 0, "neg": 0, "neu": 1},
            "textblob_polarity": 0.0,
            "textblob_subjectivity": 0.0,
        }

    # VADER analysis
    vader_scores = vader_analyzer.polarity_scores(text)

    # TextBlob analysis (type ignored due to Pyright cached_property false positive)
    blob = TextBlob(text)
    polarity = float(blob.sentiment.polarity)  # type: ignore
    subjectivity = float(blob.sentiment.subjectivity)  # type: ignore

    # Granular classification
    emotion, intensity, category = _classify_granular_emotion(vader_scores, text)

    return {
        "emotion": emotion,
        "intensity": intensity,
        "category": category,
        "vader_scores": {
            "compound": round(vader_scores["compound"], 4),
            "pos": round(vader_scores["pos"], 4),
            "neg": round(vader_scores["neg"], 4),
            "neu": round(vader_scores["neu"], 4),
        },
        "textblob_polarity": round(polarity, 4),
        "textblob_subjectivity": round(subjectivity, 4),
    }


if __name__ == "__main__":
    # Quick test
    test_texts = [
        "I am so happy and excited about this amazing opportunity!",
        "This is terrible and frustrating, nothing works properly.",
        "I'm not sure about this, it makes me worried.",
        "The weather is okay today.",
        "Wow, I can't believe this actually happened!",
        "Why is this not working? I need help understanding the issue.",
        "I feel so sad and lonely after what happened.",
    ]

    for text in test_texts:
        result = detect_emotion(text)
        print(f"\nText: {text}")
        print(f"  Emotion: {result['emotion']} ({result['category']})")
        print(f"  Intensity: {result['intensity']}")
        print(f"  VADER compound: {result['vader_scores']['compound']}")
