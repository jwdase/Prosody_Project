import torch
import numpy as np

from torch.utils.data import DataLoader
from language_detection.utils.io import check_path, save_spect
from language_detection import config
from language_detection.data.spectrogram.dataset import AudioFileDataset

def collate_fn(batch):
    """
    Takes in a batch of [1 length] audio files and then stacks them
    all audio files must be the same length
    """
    group = torch.stack(batch)

    return group


def lang_use_script(lang, dataset, base, process):
    """
    Runs the parsing code, where spectrograms are
    created and saved as batches

    process to be a tuple: (sr, n_ftt, hop_length)
    """

    # Loads the data files
    loader = DataLoader(
        dataset, batch_size=128, collate_fn=collate_fn, num_workers=8, drop_last=True
    )

    window = config.WINDOW(process["n_fft"], device=config.DEVICE)

    i = 0
    for batch_waveforms in loader:
        specs = process['spect_f'](
            lang, batch_waveforms, window, process["n_fft"], process["hop_length"], process['sr'],
        )
        save_spect(specs, base, len(specs), i)
        i += 1


def make_spect(lang, data, base, audio_process, samples):
    """
    Code that runs to make a new spectrogram
    """

    check_path(base)

    inputs = np.random.choice(data, size=samples, replace=False)

    dataset = AudioFileDataset(inputs, audio_process["sr"], audio_process['length'])

    # Creates spectrogram and saves files
    lang_use_script(
        lang,
        dataset,
        base,
        audio_process,
    )