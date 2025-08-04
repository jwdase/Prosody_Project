"""
Script to make normal spectrograms - specifies
the attributes decided by main
"""

import torch

from path import setup_project_root

setup_project_root()

from langauge_detection.data.spectrograms import main as make_spect

def compute_spectrogram_batch(lang, batch, window, n_fft, hop_length):
    """
    Takes a batch moves it to the GPU then gives the
    """
    batch = batch.to("cuda")

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


def main(languages, time_frame, num_speakers, audio_process, new_location):
    """
    Code to run spectrogram creation
    """
    make_spect(
        languages,
        time_frame,
        num_speakers,
        audio_process,
        new_location,
        compute_spectrogram_batch,
    )

if __name__ == '__main__':
    languages = ["en", "es", "de", "it"]
    window = "5.5 - 6.0"

    location = "/om2/user/moshepol/prosody/data/raw_audio/"

    n_ftt = 1024
    hop_length = 512
    sr = 16_000

    entry = {"sr": sr, "n_fft": n_ftt, "hop_length": hop_length}

    main(languages, window, 2, entry, location)

