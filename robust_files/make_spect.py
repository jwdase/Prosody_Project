"""
This script uses the CPU to create new spectrograms
for the audio files, it splits them into training, testing,
and validation, there is no speaker crossover. Files will be placed into
/om2/user/moshepol/prosody/data/raw_audio/{lang}/spect/{use}/{placement}/

lang: language
use: train, test, validation
placement: time domain range
"""

from path import setup_project_root
setup_project_root()

import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio


from langaugedetection.data.length_df_tools import select_array
from langaugedetection.data.spectrogram import parse


def select_language_time(languages, window):
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

        lang_to_options[lang] = select_array(df, window)

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

    max_length = 1_000_000_000

    for lang, item in lang_dict.items():
        max_length = min(max_length, len(item))

    partition = (max_length // 1_000) * 1_000

    # Overide, Make sure to remove when submitting it
    partition = 10_000

    train = int(partition * 0.8)
    test = int(partition * 0.1)

    return train, train + test, train + test * 2


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


def split_dataset(files, train, test, val):
    """
    Splits the dataset into it's train, test, val
    """

    # Takes the data randomly
    data = np.random.choice(files, size=(train + test + val), replace=False)

    train_files = data[0:train]
    test_files = data[train:test]
    validation_files = data[test:val]

    return train_files, test_files, validation_files


def save_files(spectrograms, path, batch_size, i):
    """
    Code to save the files,
    """

    def file_name(length, num):
        return "batch_" + '0' * (length - len(str(num))) + str(num) + '_' + str(batch_size)

    torch.save(spectrograms, Path(path + "/" + file_name(5, i) + '.pt'))


class AudioFileDataset(Dataset):
    def __init__(self, audio_dir, max_time=5.5, sr=16_000):
        self.files = list(audio_dir)
        self.target_sr = sr
        self.length = max_time * sr
        self.samplers = {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        waveform, sr = torchaudio.load(path)

        if sr != self.target_sr:

            if sr not in self.samplers.keys():
                self.samplers[sr] = torchaudio.transforms.Resample(sr, self.target_sr)
                
            new_wave = self.samplers[sr](waveform)
            return self.pad_waveform(new_wave)

        return self.pad_waveform(waveform)

    def pad_waveform(self, waveform):
        length = waveform.shape[-1]

        if length > self.length:
            raise TypeError

        else:
            pad = self.length - length
            return torch.nn.functional.pad(waveform, (0, pad))


def collate_fn(batch):
    '''
    Takes in a batch of [1 length] audio files and then stacks them
    all audio files must be the same length
    '''
    group = torch.stack(batch)

    return group

def compute_spectrogram_batch(batch, window, n_fft=1024, hop_length=256):
    batch = batch.to('cuda')

    specs = torch.stft(
        batch.squeeze(1),
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
        window=window
    )

    power = specs.abs() ** 2
    db = 10 * torch.log10(torch.clamp(power, min=1e-10))

    return db.cpu()  


def lang_use_script(files, base):
    '''
    Runs the parsing code, where spectrograms are
    created and saved as batches
    '''

    # Creates the dataset
    dataset = AudioFileDataset(files)

    # Loads the data files
    loader = DataLoader(
        dataset,
        batch_size=128,
        collate_fn=collate_fn,
        num_workers=8,
        drop_last=True
    )

    window = torch.hann_window(1024, device='cuda')

    i = 0 
    for batch_waveforms in loader:
        specs = compute_spectrogram_batch(batch_waveforms, window)
        save_files(specs, base, len(specs), i)
        i += 1


def main(languages, window):
    '''
    Pulls the dataframes and runs the main script
    '''
    # Dictionary
    lang_dict = select_language_time(languages, window)

    # Folder Name, Placement of datasets
    placement = "range_" + window.replace(".", "_").replace(" ", "")
    data_names = ("train", "test", "validation")

    # Specifies indices for each split
    split = train_test_val_split(lang_dict)


    for lang, files in lang_dict.items():
        datasets = {
            data_type : data
            for data_type, data in zip(data_names, split_dataset(files, *split))
        }

        for use, data in datasets.items():
            print(f'Language: {lang}, Use: {use}, Data Size: {len(data)}')

            # Ensure directory is working
            base = base = f"/om2/user/moshepol/prosody/data/raw_audio/{lang}/spect/{use}/{placement}/"
            check_path(base)

            # Creates spectrogram and saves files
            lang_use_script(data, base)

        print(f'Completed {lang} with this path: {base}')



if __name__ == '__main__':
    # In train, Val, Split
    # Overide partition if you want this to work 
    # On the whole dataset


    languages = ["en", "it", "es", "de"]
    window = "5.5 - 6.0"

    main(languages, window)

    print('Done')