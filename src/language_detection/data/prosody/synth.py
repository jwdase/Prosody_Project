"""Prosody-only audio synthesis.

Python port of the MATLAB pipeline in ``src/matlab/synthOnlyProsody.m`` (Tamar
Regev, 2023). It strips everything but prosody (the pitch contour and the
loudness contour) from a speech recording, producing a buzzy tone that follows
the speech intonation and rhythm but carries no phonetic / timbral content.

The original relied on TANDEM-STRAIGHT. Here STRAIGHT is replaced by the WORLD
vocoder (``pyworld``) - its maintained, pip-installable successor from the same
research lineage. WORLD provides the same analysis/synthesis primitives:
F0 estimation (Harvest), spectral envelope (CheapTrick) and band aperiodicity
(D4C).

Pipeline (mirrors the MATLAB version):
  1. Estimate the speech F0 contour.
  2. Compute the speech loudness envelope (rectify -> low-pass -> sqrt).
  3. Generate a fixed harmonic complex tone of the same duration.
  4. Analyse the tone to get a speech-independent timbre (spectral envelope +
     aperiodicity).
  5. Impose the speech F0 contour on the tone and resynthesise.
  6. Modulate the result by the speech loudness envelope.
"""

from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf
from scipy.signal import butter, lfilter

# Synthesis tone defaults (match the MATLAB script).
TONE_F0 = 200.0
TONE_HARMONICS = 100
TONE_ATTEN_DB = -12.0  # spectral roll-off per octave (dist == 0 branch)
SMOOTH_SEC = 0.125  # "FAST" SPL integration time -> 8 Hz low-pass
RAMP_MS = 10.0
FRAME_PERIOD = 5.0  # WORLD analysis/synthesis frame shift, ms
OUTPUT_GAIN = 5.0


def _hann_ramp(x, ramp_ms, sr):
    """Apply a raised-cosine fade in/out to remove onset/offset clicks."""
    ramp = int(np.floor(ramp_ms * sr / 1000.0))
    if len(x) < 2 * ramp:
        raise ValueError("Ramps cannot be longer than the stimulus duration")
    win = np.hanning(2 * ramp)
    out = x.copy()
    out[:ramp] *= win[:ramp]
    out[-ramp:] *= win[ramp:]
    return out


def generate_complex_tone(f0, harmonics, dur_s, sr, atten_db=TONE_ATTEN_DB):
    """Harmonic complex tone with an exponential spectral envelope.

    Port of ``generate_singlenote_vary_envelope_jitter_randphase_Tamar.m`` with
    ``jitt == 0`` (harmonic) and ``dist == 0`` (exponential decay, flat
    amplitude envelope) - the settings used by ``synthOnlyProsody.m``.
    """
    t = np.arange(1, round(dur_s * sr) + 1) / sr
    freqs = f0 * np.arange(1, harmonics + 1, dtype=float)
    freqs[freqs > sr / 2] = 0.0  # zero out components above Nyquist

    harm = np.sin(2 * np.pi * np.outer(freqs, t))
    spec_env = 10.0 ** (atten_db * np.log2(np.arange(1, len(freqs) + 1)) / 20.0)
    s = (spec_env[:, None] * harm).sum(axis=0)
    return _hann_ramp(s, RAMP_MS, sr)


def loudness_envelope(y, fs, smooth_sec=SMOOTH_SEC):
    """Speech amplitude (loudness) trajectory.

    Rectify by squaring, low-pass with a 1st-order Butterworth filter, then take
    the square root - the SPL-style envelope from the MATLAB script. Uses a
    causal filter (``lfilter``) to match MATLAB's ``filter``.
    """
    smooth_hz = 1.0 / smooth_sec
    b, a = butter(1, smooth_hz / (fs / 2.0), btype="low")
    return np.sqrt(np.maximum(lfilter(b, a, y ** 2), 0.0))


def _interp_unvoiced(f0):
    """Linearly interpolate F0 across unvoiced (zero) frames.

    STRAIGHT carries a continuous F0 trajectory and gates silence via the
    loudness envelope. Harvest reports 0 in unvoiced frames, so we interpolate
    to keep the synthesised tone continuous; the loudness envelope then silences
    the gaps.
    """
    voiced = f0 > 0
    if not voiced.any():
        return f0
    idx = np.arange(len(f0))
    return np.interp(idx, idx[voiced], f0[voiced])


def synth_only_prosody(input_path, output_path, frame_period=FRAME_PERIOD):
    """Synthesise a prosody-only rendering of a speech file and write it out.

    Args:
        input_path: speech audio readable by libsndfile (wav, flac, ...).
        output_path: destination .wav path.
        frame_period: WORLD frame shift in ms.
    """
    y, fs = sf.read(str(input_path), dtype="float64", always_2d=False)
    if y.ndim > 1:  # mix down to mono
        y = y.mean(axis=1)

    # 1. speech F0 contour
    f0_speech, t = pw.harvest(y, fs, frame_period=frame_period)
    f0_speech = pw.stonemask(y, f0_speech, t, fs)

    # 2. speech loudness envelope
    env = loudness_envelope(y, fs)

    # 3. fixed-timbre complex tone, matched in duration
    dur_s = len(y) / fs
    tone = generate_complex_tone(TONE_F0, TONE_HARMONICS, dur_s, fs)

    # 4. analyse the tone -> speech-independent spectral envelope + aperiodicity
    f0_tone, t_tone = pw.harvest(tone, fs, frame_period=frame_period)
    f0_tone = pw.stonemask(tone, f0_tone, t_tone, fs)
    sp = pw.cheaptrick(tone, f0_tone, t_tone, fs)
    ap = pw.d4c(tone, f0_tone, t_tone, fs)

    # 5. impose the speech pitch on the tone timbre and resynthesise
    n = min(len(f0_speech), sp.shape[0])
    f0 = _interp_unvoiced(f0_speech[:n])
    out = pw.synthesize(f0, sp[:n], ap[:n], fs, frame_period=frame_period)

    # 6. normalise, impose loudness, scale, write
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.8
    m = min(len(out), len(env))
    out = out[:m] * env[:m] * OUTPUT_GAIN

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), out, fs)
