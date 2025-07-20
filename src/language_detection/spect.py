"""
Tools to load and compute a spectrogram on a GPU
"""

import torch

from language_detection.data.spectrogram.loader import (
    select_language_time,
    train_test_val_split,
)

from language_detection.data.spectrogram.compute import make_spect
from language_detection.data.spectrogram.functions import compute_spectrogram_batch, compute_lowpass_spectrogram_batch


def main(languages, time_frame, audio_process, new_location):
    """
    Pulls the dataframe and makes the spectrogram
    """
    lang_dict = select_language_time(languages, time_frame)

    # Folder Name, Placement of datasets
    placement = "range_" + time_frame.replace(".", "_").replace(" ", "")

    sizes = train_test_val_split(lang_dict)

    for lang, datasets in lang_dict.items():
        for use, data in datasets.items():

            # Ensure directory is working
            base = base = f"{new_location}/{lang}/spect/{use}/{placement}/"

            samples = int((sizes[use] // 100) * 100)

            make_spect(lang, data, base, audio_process, samples)
        
            print(f'Completed: {lang}, Size: {samples}')

if __name__ == '__main__':
    languages = ["en", "es", "it", 'de']
    window = "5.5 - 6.0"

    location = "/om2/user/moshepol/prosody/data/low_pass/"

    n_ftt = 1024
    hop_length = 512
    sr = 16_000

    entry = {"sr": sr, "n_fft": n_ftt, "hop_length": hop_length, "length" : 6.0, "spect_f" : compute_lowpass_spectrogram_batch}

    main(languages, window, entry, location)

