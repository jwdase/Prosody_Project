
import torch
import pandas as pd

from language_detection.data.csv_tools.read import select_array
from language_detection import config

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

        lang_to_options[lang] = select_array(df, window, config.NUM_SPEAKERS)

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
