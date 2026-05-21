"""Musical key helpers used by the API, CLI, and inference pipeline."""

from __future__ import annotations

KEY_NAMES = ("c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b")

FLAT_TO_SHARP = {
    "db": "c#",
    "eb": "d#",
    "gb": "f#",
    "ab": "g#",
    "bb": "a#",
}


def normalize_key(value: str) -> str:
    key = value.strip().lower().replace("♯", "#").replace("♭", "b")
    key = FLAT_TO_SHARP.get(key, key)
    if key not in KEY_NAMES:
        valid = ", ".join(KEY_NAMES)
        raise ValueError(f"Unknown key '{value}'. Use one of: {valid}")
    return key


def key_to_index(value: str) -> int:
    return KEY_NAMES.index(normalize_key(value))


def key_name(index: int) -> str:
    return KEY_NAMES[index % len(KEY_NAMES)]


def shortest_semitone_shift(source_key_index: int, target_key_index: int) -> int:
    """Return the shortest signed semitone shift from source to target."""

    return (target_key_index - source_key_index + 6) % 12 - 6

