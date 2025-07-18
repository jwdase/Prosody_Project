"""
Tools to load and compute a spectrogram on a GPU
"""

import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

from .length_df_tools import select_array

LENGTH = 5.5  # Length of Audio Files in Seconds


def select_language_time(languages, window, num_speakers):
    """
    Given a list of languages, and their respective time
    window loads a .csv containing languages and audio files
    correlating to that window. It returns a dictionary with
    key : language
    item : list of audio files
    """
    lang_to_options = {}

    for lang in languages:
        path = f"/om2/user/moshepol/prosody/data/raw_audio/{lang}/custom/length.csv"
        df = pd.read_csv(path)

        lang_to_options[lang] = select_array(df, window, num_speakers)

        print(f"Finished reading: {lang}")

    return lang_to_options


def train_test_val_split(lang_dict):
    """
    Returns the indices for how the data should be
    split between training and testing - does this with
    80% train
    10% test
    10% validation
    """

    x = 1_000_000_000

    max_size = {"train": x, "test": x, "val": x}

    for lang, dataset in lang_dict.items():
        for dtype, data in dataset.items():
            max_size[dtype] = min(max_size[dtype], len(data))

    return max_size


def check_path(base):
    """
    Clears the folder that we're going to
    laod the spectrograms into
    """
    folder = Path(base)

    # Checks if the folder exists, and then deletes folder and recreates it
    if folder.is_dir():
        shutil.rmtree(base)

    folder.mkdir(exist_ok=True, parents=True)


def save_files(spectrograms, path, batch_size, i):
    """
    Code to save the files,
    """

    def file_name(length, num):
        return (
            "batch_" + "0" * (length - len(str(num))) + str(num) + "_" + str(batch_size)
        )

    torch.save(spectrograms, Path(path + "/" + file_name(5, i) + ".pt"))


class AudioFileDataset(Dataset):
    """
    We have to use a Dataset to load the Audio signal
    onto the GPU, then the Spectrogram computation occurs
    on the GPU
    """

    def __init__(self, audio_dir, sr, max_time):
        self.files = list(audio_dir)
        self.target_sr = sr
        self.length = int(max_time * sr)
        self.samplers = {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        waveform, sr = torchaudio.load(path)

        if sr != self.target_sr:

            if sr not in self.samplers:
                self.samplers[sr] = torchaudio.transforms.Resample(sr, self.target_sr)

            new_wave = self.samplers[sr](waveform)
            return self.pad_waveform(new_wave)

        return self.pad_waveform(waveform)

    def pad_waveform(self, waveform):
        length = waveform.shape[-1]

        if length > self.length:
            raise TypeError

        pad = self.length - length
        return torch.nn.functional.pad(waveform, (0, pad))


def collate_fn(batch):
    """
    Takes in a batch of [1 length] audio files and then stacks them
    all audio files must be the same length
    """
    group = torch.stack(batch)

    return group


def lang_use_script(lang, files, base, process, spect_func, length):
    """
    Runs the parsing code, where spectrograms are
    created and saved as batches

    process to be a tuple: (sr, n_ftt, hop_length)
    """

    # Creates the dataset
    dataset = AudioFileDataset(files, process["sr"], length)

    # Loads the data files
    loader = DataLoader(
        dataset, batch_size=128, collate_fn=collate_fn, num_workers=8, drop_last=True
    )

    window = torch.hann_window(process["n_fft"], device="cuda")

    i = 0
    for batch_waveforms in loader:
        specs = spect_func(
            lang, batch_waveforms, window, process["n_fft"], process["hop_length"]
        )
        save_files(specs, base, len(specs), i)
        i += 1


def main(languages, time_frame, num_speakers, audio_process, new_location, spect_func):
    """
    Pulls the dataframes and runs the main script,

    audio_proces is a tuple: (sr, n_ftt, hop_length)
    """
    # Dictionary
    lang_dict = select_language_time(languages, time_frame, num_speakers)

    # Folder Name, Placement of datasets
    placement = "range_" + time_frame.replace(".", "_").replace(" ", "")

    sizes = train_test_val_split(lang_dict)

    for lang, datasets in lang_dict.items():
        for use, data in datasets.items():
            # Ensure directory is working
            base = base = f"{new_location}/{lang}/spect/{use}/{placement}/"
            check_path(base)

            # Clip to Maximum Size
            file_num = (sizes[use] // 100) * 100

            print(f"Language: {lang}, Use: {use}, Data Size: {file_num}")

            # Randomly select from files
            inputs = np.random.choice(data, size=file_num, replace=False)

            # Creates spectrogram and saves files
            lang_use_script(
                lang,
                inputs,
                base,
                audio_process,
                spect_func,
                float(time_frame.split(" ")[0]),
            )

        print(f"Completed {lang} with this path: {base}")

