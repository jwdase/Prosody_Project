import random

import torch
import torchaudio

from language_detection import config
from language_detection.utils.io import check_path, last_audio, make_name

def choices(lang, audio, num_samples, directory, sr):
    """
    Saves the audio file under the directory specified
    """
    directory = f"{config.AUDIO_SAVED}/{lang}/recordings"

    # Ensures Directory Exists
    check_path(directory)

    saved_files = random.sample(range(audio.size(0)), num_samples)

    start = last_audio(directory)

    for name, i in enumerate(saved_files):
        path = f"{directory}/{make_name(start + name + 1)}"
        torchaudio.save(path, audio[i].cpu(), sr)


def group_by_lang(data):
    """
    Takes in a dataset from create_dataset 

    Structure:
        {'lang' : {'type' : {time : audio files}}}

    And creates a dataset of type

    Structure:
        {'lang' : {'type' : [ Files ]}} 
    """

    x = {}

    for lang, datasets in data.items():
        x[lang] = {}

        for use, data in datasets.items():
            x[lang][use] = []
            for buckey, files in data.items():
                x[lang][use] += files

    return x