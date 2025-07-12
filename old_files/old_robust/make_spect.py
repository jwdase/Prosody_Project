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
import torch

from path import setup_project_root
setup_project_root()


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
    partition = 3_000

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
    train_files = files[0:train]
    test_files = files[train:test]
    validation_files = files[test:val]

    return train_files, test_files, validation_files


def save_files(spectrograms, path):
    """
    Code to save the files
    """

    def file_name(length, num):
        return "0" * (length - len(str(num))) + str(num)

    for i, spect in enumerate(spectrograms):
        torch.save(torch.tensor(spect), Path(path + "/" + file_name(6, i) + '.pt'))


def main(languages, window, length):
    """
    Code where we load audio, make spect, and place
    in respective folder
    """
    # Dictionary
    lang_dict = select_language_time(languages, window)

    # Folder Name, Placement of datasets
    placement = "range_" + window.replace(".", "_").replace(" ", "")
    data_names = ("train", "test", "validation")

    # Specifies indices for each split
    split = train_test_val_split(lang_dict)

    for lang, files in lang_dict.items():
        datasets = {
            data_type: data
            for data_type, data in zip(data_names, split_dataset(files, *split))
        }

        for use, data in datasets.items():
            # Makes Spectrograms
            spectrograms = parse(data, length)

            # Ensures the directory is clear and working
            base = f"/om2/user/moshepol/prosody/data/raw_audio/{lang}/spect/{use}/{placement}/"
            check_path(base)

            # Saves the files
            save_files(spectrograms, base)

        print(f"Completed {lang} with this path: {base}")


if __name__ == "__main__":
    languages = ["en", "it", "es", "de"]
    window = "5.5 - 6.0"
    length = 46

    main(languages, window, 46)

    print("Done")