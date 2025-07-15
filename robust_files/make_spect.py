"""
This script uses the CPU to create new spectrograms
for the audio files, it splits them into training, testing,
and validation, there is no speaker crossover. Files will be placed into
/om2/user/moshepol/prosody/data/raw_audio/{lang}/spect/{use}/{placement}/

lang: language
use: train, test, validation
placement: time domain range
"""


import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

from path import setup_project_root
setup_project_root()

from langaugedetection.data.length_df_tools import select_array

LENGTH = 5.5  # Length of Audio Files in Seconds
WINDOW = "5.5 - 6.0"  # Specifies Window Size


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
    def __init__(self, audio_dir, sr, max_time=LENGTH):
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


def compute_spectrogram_batch(batch, window, n_fft, hop_length):
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


def lang_use_script(files, base, process):
    """
    Runs the parsing code, where spectrograms are
    created and saved as batches

    process to be a tuple: (sr, n_ftt, hop_length)
    """

    # Creates the dataset
    dataset = AudioFileDataset(files, process["sr"])

    # Loads the data files
    loader = DataLoader(
        dataset, batch_size=128, collate_fn=collate_fn, num_workers=8, drop_last=True
    )

    window = torch.hann_window(process["n_fft"], device="cuda")

    i = 0
    for batch_waveforms in loader:
        specs = compute_spectrogram_batch(
            batch_waveforms, window, process["n_fft"], process["hop_length"]
        )
        save_files(specs, base, len(specs), i)
        i += 1


def main(languages, time_frame, num_speakers, audio_process, new_location):
    """
    Pulls the dataframes and runs the main script,

    audio_proces is a tuple: (sr, n_ftt, hop_length)
    """
    # Dictionary
    lang_dict = select_language_time(languages, time_frame, num_speakers)

    # Folder Name, Placement of datasets
    placement = "range_" + time_frame.replace(".", "_").replace(" ", "")

    sizes = train_test_val_split(lang_dict)
    print(sizes)

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
            lang_use_script(inputs, base, audio_process)

        print(f"Completed {lang} with this path: {base}")


if __name__ == "__main__":
    # In train, Val, Split
    # Overide partition if you want this to work
    # On the whole dataset

    languages = ["en", "es", "de"]
    window = WINDOW

    location = "/om2/user/moshepol/prosody/data/raw_audio/"

    n_ftt = 1024
    hop_length = 512
    sr = 16_000

    entry = {"sr": sr, "n_fft": n_ftt, "hop_length": hop_length}

    main(languages, window, 2, entry, location)

    # print("Done")
