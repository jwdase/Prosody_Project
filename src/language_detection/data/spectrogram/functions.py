import random
import glob

import torch
import torchaudio

from language_detection import config
from language_detection.utils.io import check_path, last_audio, make_name
from language_detection.data.spectrogram.tools import choices

def compute_spectrogram_batch(lang, batch, window, n_fft, hop_length, sr):
    """
    Takes a batch moves it to the GPU then gives the
    """
    batch = batch.to(config.DEVICE)

    specs = torch.stft(
        batch.squeeze(1),
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
        window=window,
    )

    power = specs.abs() ** 2
    db = 10 * torch.log10(torch.clamp(power, min=1e-10))

    return db.cpu()


def compute_lowpass_spectrogram_batch(lang, batch, window, n_fft, hop_length, sr):
    """
    Applies a low pass filer to a batch of audio
    and randomly saves 10% of audio files for reference later
    it then turns the audio into spectrograms
    """
    batch = batch.to(config.DEVICE)

    # Filter Audio
    filtered_audio, mask = fft_lowpass_batch(batch, sr, config.CUTOFF)

    # Saves some audio files
    choices(lang, filtered_audio, 2, config.AUDIO_SAVED, sr)

    # Retursn complex
    specs = torch.stft(
        filtered_audio.squeeze(1),
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
        window=window,
    )

    power = specs.abs() ** 2
    db = 10 * torch.log10(torch.clamp(power, min=1e-10))

    # Removes the frequency bins that are above our cutoff
    freqs = torch.fft.rfftfreq(n_fft, 1 / sr).to(config.DEVICE)
    mask = freqs <= config.CUTOFF
    db_trimmed = db[:, mask, :]

    return db_trimmed.cpu()


def fft_lowpass_batch(audio, sr, cutoff):
    """
    Applies a sharp FFT based low-pass filter to audio

    Args:
        audio : Tensor of shape [B, 1, T]
        sr: audio sample rate 16,000
        cutoff: where we want the low pass cutoff to end

    """
    audio = audio.squeeze(1)  # [B, 1, T] --> [B, T]

    # FFT
    fft = torch.fft.rfft(audio, dim=1)
    freqs = torch.fft.rfftfreq(audio.shape[-1], d=1 / sr).to(config.DEVICE)

    # Creates a mask and applies it
    mask = freqs <= cutoff
    fft = fft * mask

    # Turn back into audio
    audio = torch.fft.irfft(fft, n=audio.shape[-1], dim=-1).unsqueeze(1)  # [B, 1, T]
    return audio, mask






