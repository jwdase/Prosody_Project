"""Prosody-only audio synthesis (Python port of the MATLAB STRAIGHT pipeline)."""

from language_detection.data.prosody.synth import (
    generate_complex_tone,
    loudness_envelope,
    synth_only_prosody,
)

__all__ = [
    "generate_complex_tone",
    "loudness_envelope",
    "synth_only_prosody",
]
