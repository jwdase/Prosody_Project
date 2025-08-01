import torch
import numpy as np

from torch.utils.data import DataLoader
from language_detection.utils.io import save_spect
from language_detection import config
from language_detection.data.spectrogram.dataset import AudioFileDataset
from language_detection import config

def collate_fn(batch):
    """
    Takes in a batch of [1 length] audio files and then stacks them
    all audio files must be the same length
    """
    waveforms, lengths = zip(*batch)

    return torch.stack(waveforms), torch.tensor(lengths, dtype=torch.long)


def lang_use_script(lang, dataset, base, process):
    """
    Runs the parsing code, where spectrograms are
    created and saved as batches

    process to be a tuple: (sr, n_ftt, hop_length)
    """

    # Loads the data files
    loader = DataLoader(
        dataset, batch_size=config.SPECT_SIZE, collate_fn=collate_fn, num_workers=config.WORKERS, drop_last=True
    )

    window = config.WINDOW(process["n_fft"], device=config.DEVICE)

    i = 0
    for batch_waveforms, batch_lengths in loader:
        specs = process['spect_f'](
            lang, batch_waveforms, window, process["n_fft"], process["hop_length"], process['sr'],
        )

        # Saves with length mapping
        files = {
            'spec' : specs,
            'length' : (batch_lengths - process["n_fft"]) // process['hop_length'] + 1
        }

        save_spect(files, base, len(specs), i)
        i += 1


def make_spect(lang, inputs, base, audio_process):
    """
    Code that runs to make a new spectrogram
    """

    dataset = AudioFileDataset(inputs, audio_process["sr"], config.MAX_LENGTH)

    # Creates spectrogram and saves files
    lang_use_script(
        lang,
        dataset,
        base,
        audio_process,
    )