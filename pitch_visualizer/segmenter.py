"""
Narrative Segmentation Module for The Pitch Visualizer.
Breaks narrative text into logical scenes using NLTK sentence tokenization.
"""

import re
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag


def extract_key_phrases(sentence: str) -> dict:
    """
    Extract key nouns, verbs, and adjectives from a sentence.

    Returns dict with keys: nouns, verbs, adjectives, entities
    """
    try:
        words = word_tokenize(sentence)
        tagged = pos_tag(words)
    except Exception:
        return {"nouns": [], "verbs": [], "adjectives": [], "entities": []}

    nouns = []
    verbs = []
    adjectives = []
    entities = []

    for word, tag in tagged:
        if len(word) < 2:
            continue
        if tag.startswith('NN'):
            nouns.append(word.lower())
        elif tag.startswith('VB') and tag != 'VBZ':
            verbs.append(word.lower())
        elif tag.startswith('JJ'):
            adjectives.append(word.lower())
        elif tag == 'NNP':
            entities.append(word)

    return {
        "nouns": list(set(nouns))[:5],
        "verbs": list(set(verbs))[:3],
        "adjectives": list(set(adjectives))[:3],
        "entities": list(set(entities))[:3],
    }


def _detect_topic_shift(sent1: str, sent2: str) -> bool:
    """Detect if there's a topic shift between two sentences."""
    # Simple heuristic: check for transition words or significant noun changes
    transition_words = [
        "however", "but", "meanwhile", "then", "suddenly", "finally",
        "next", "afterwards", "later", "eventually", "consequently",
        "furthermore", "moreover", "in contrast", "on the other hand",
        "as a result", "therefore", "thus"
    ]

    sent2_lower = sent2.lower()
    for tw in transition_words:
        if sent2_lower.startswith(tw) or f" {tw} " in sent2_lower:
            return True

    # Check noun overlap — low overlap suggests topic shift
    kp1 = extract_key_phrases(sent1)
    kp2 = extract_key_phrases(sent2)

    nouns1 = set(kp1["nouns"])
    nouns2 = set(kp2["nouns"])

    if nouns1 and nouns2 and len(nouns1 & nouns2) == 0:
        return True

    return False


def segment_text(text: str, min_segments: int = 3, max_segments: int = 6) -> list:
    """
    Segment narrative text into logical scenes.

    Args:
        text: Input narrative text
        min_segments: Minimum number of segments (default 3)
        max_segments: Maximum number of segments (default 6)

    Returns:
        List of dicts, each with:
            - id: int (1-indexed)
            - text: str (the segment text)
            - sentences: list of str
            - key_phrases: dict with nouns, verbs, adjectives
            - position: str ("opening", "middle", "climax", "closing")
    """
    if not text or not text.strip():
        return []

    # Tokenize into sentences
    sentences = sent_tokenize(text.strip())

    if len(sentences) <= min_segments:
        # Each sentence is its own segment
        segments = []
        for i, sent in enumerate(sentences):
            position = _get_position(i, len(sentences))
            kp = extract_key_phrases(sent)
            segments.append({
                "id": i + 1,
                "text": sent,
                "sentences": [sent],
                "key_phrases": kp,
                "position": position,
            })
        return segments

    # Group sentences into segments based on topic shifts
    groups = [[sentences[0]]]

    for i in range(1, len(sentences)):
        if len(groups) < max_segments and _detect_topic_shift(sentences[i - 1], sentences[i]):
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])

    # If we have fewer than min_segments, split the largest group
    while len(groups) < min_segments and any(len(g) > 1 for g in groups):
        # Find largest group
        largest_idx = max(range(len(groups)), key=lambda x: len(groups[x]))
        if len(groups[largest_idx]) <= 1:
            break
        group = groups[largest_idx]
        mid = len(group) // 2
        groups[largest_idx] = group[:mid]
        groups.insert(largest_idx + 1, group[mid:])

    # If still fewer than min_segments and we have enough sentences, force split
    if len(groups) < min_segments:
        flat_sentences = [s for g in groups for s in g]
        chunk_size = max(1, len(flat_sentences) // min_segments)
        groups = []
        for i in range(0, len(flat_sentences), chunk_size):
            groups.append(flat_sentences[i:i + chunk_size])
            if len(groups) >= max_segments:
                # Add remaining to last group
                if i + chunk_size < len(flat_sentences):
                    groups[-1].extend(flat_sentences[i + chunk_size:])
                break

    # Build segment objects
    segments = []
    for i, group in enumerate(groups):
        segment_text_joined = " ".join(group)
        position = _get_position(i, len(groups))
        kp = extract_key_phrases(segment_text_joined)
        segments.append({
            "id": i + 1,
            "text": segment_text_joined,
            "sentences": group,
            "key_phrases": kp,
            "position": position,
        })

    return segments


def _get_position(index: int, total: int) -> str:
    """Determine narrative position label."""
    if total <= 1:
        return "opening"
    if index == 0:
        return "opening"
    elif index == total - 1:
        return "closing"
    elif index == total - 2 or (total > 3 and index >= total * 0.6):
        return "climax"
    else:
        return "middle"


if __name__ == "__main__":
    test_text = """
    Our client, a Fortune 500 company, was struggling with declining customer engagement.
    Their legacy systems were outdated and their team was frustrated with the lack of innovation.
    We stepped in with our cutting-edge AI solution that transformed their customer experience.
    Within three months, customer satisfaction scores increased by 45%.
    The CEO personally thanked our team for the remarkable turnaround.
    Today, they are one of our most successful case studies and a testament to innovation.
    """

    segments = segment_text(test_text)
    for seg in segments:
        print(f"\n--- Scene {seg['id']} ({seg['position']}) ---")
        print(f"Text: {seg['text']}")
        print(f"Key phrases: {seg['key_phrases']}")
