"""
Tools to load and compute a spectrogram on a GPU
"""

import pickle

import torch

from language_detection.data.spectrogram.loader import create_dataset
from language_detection.data.spectrogram.compute import make_spect
from language_detection.data.spectrogram.functions import compute_spectrogram_batch, compute_lowpass_spectrogram_batch
from language_detection.data.spectrogram.tools import group_by_lang
from language_detection.utils.io import check_path


def main(languages, audio_process, new_location):
    """
    Pulls the dataframe and makes the spectrogram of the
    audio files
    """

    # Loads dataset by partition
    dataset, speakers = create_dataset(languages)

    # Saves the 2 files
    with open(f'{new_location}/dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    with open(f'{new_location}/speakers.pkl', 'wb') as f:
        pickle.dump(speakers, f)

    # Cleans array
    dataset = group_by_lang(dataset)

    for lang, dataset in dataset.items():
        print(f'Starting Spectrogram: {lang}')
        for use, data in dataset.items():

            # Cleans and then writes to directory
            base = f"{new_location}/{lang}/spect/{use}/"
            check_path(base)

            make_spect(lang, data, base, audio_process)

            print(f"Finished: {use} w/ {len(data)} samples")

        print(f"Finsihed: {lang}")


if __name__ == '__main__':
    languages = ["en", "it", "es", "de", "nl", "ta", "ja"]

    location = "/om2/user/moshepol/prosody/data/low_pass_data"

    n_ftt = 1024
    hop_length = 512
    sr = 16_000

    entry = {"sr": sr, "n_fft": n_ftt, "hop_length": hop_length, "spect_f" : compute_lowpass_spectrogram_batch}

    main(languages, entry, location)

