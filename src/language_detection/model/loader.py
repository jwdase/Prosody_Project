import glob

import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from language_detection import config
from language_detection.model.dataset import AudioDataset


def load_audio(lang_list, orig, enc):
    """
    Loads the audio files from the directory specified below
    stores them as a dict
    """
    # Create the array which saves data-values
    # References allows us to manually verify the audio is
    # from the correct language
    data_types = ["train", "test", "val"]

    datasets = {dtype: AudioDataset() for dtype in data_types}

    for lang in lang_list:
        for dtype in data_types:

            paths = glob.glob(f'{orig}/{lang}/spect/{dtype}/*.pt')

            for path in paths:
                x = torch.load(path, weights_only=False)

                for spec, length in zip(x["spec"], x["length"]):
                    datasets[dtype].additem(spec, length, enc.transform([lang])[0])

    return datasets


def build_loaders(datasets):
    """
    Creates out dataloaders for training

    """

    def load_data(data, shuffle):
        return DataLoader(
            data,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            shuffle=shuffle,
        )

    train_loader = load_data(datasets["train"], True)
    test_loader = load_data(datasets["test"], False)
    validation_loader = load_data(datasets["val"], False)

    return train_loader, test_loader, validation_loader


def standize_data(dataset):
    """
    Takes in data and standardizes tensors based on training data
    mean and std
    """

    mean, std = dataset["train"].getstats()

    for dtype, data in dataset.items():

        # Normalize
        data.applynormal(mean, std)

        # Print Results
        print(f"For datatype: {dtype}")
        data.print_stats()


def print_length(dataset):
    for dtype, data in dataset.items():
        print(f"{dtype}: has {len(data)} samples")


def load_tensors(languages, data_location, enc):
    """
    Loads the tensor data_loaders
    """

    data = load_audio(languages, data_location, enc)

    standize_data(data)

    print_length(data)

    return *build_loaders(data), data["train"].get_shape()
