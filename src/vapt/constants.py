"""Shared constants, matching Kong's vendored config (begin_note=A0, 88 keys, 100 fps)."""

SAMPLE_RATE = 16000
CLASSES_NUM = 88
BEGIN_NOTE = 21  # MIDI note of A0, the lowest piano key
FRAMES_PER_SECOND = 100
HOP_S = 1.0 / FRAMES_PER_SECOND
