import glob

import torch
import numpy as np

from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from language_detection import config

def load_audio(lang_list, orig):
    """
    Loads the audio files from the directory specified below
    stores them as a dict
    key: ['train']
    value: dict (key: lang, item: tensors)
    """
    data_types = ["train", "test", "val"]

    # Create the array which saves data-values
    # References allows us to manually verify the audio is
    # from the correct language
    tensors = {dtype: {} for dtype in data_types}
    reference = {dtype: {} for dtype in data_types}

    for lang in lang_list:
        for dtype in data_types:
            base = f"{orig}/{lang}/spect/{dtype}/"

            paths = glob.glob(base + "*.pt")

            first_audio = True

            spectograms_tensors = []

            for path in paths:
                x = torch.load(path, weights_only=False)
                if first_audio:
                    reference[dtype][lang] = x['spec'][0]
                    first_audio = False

                # Put Tensor on range 0 --> 1
                spectograms_tensors.append(x)

            # Saves each languages tensor
            tensors[dtype][lang] = spectograms_tensors

    return tensors, reference


def standardize_tensors(tensors):
    """
    Normalize each audio file giving them
    their respective z-score
    """

    # Calculates tensor using 'Training data'
    train_tensors = [
        tensor for lang_tensors in tensors["train"].values() for tensor in lang_tensors
    ]

    mean, std = calc_mean(train_tensors)

    # Normalize 'Training Data'
    for use, dataset in tensors.items():
        for lang, data in dataset.items():
            tensors[use][lang] = {'spec' : [], 'length' : []}

            for d in data:

                spec_x = []
                dur_x = []

                for spec, duration in zip(d['spec'], d['length']):

                    # Normalize only over working audio
                    real = (spec[..., :duration] - mean) / std

                    # Puts in normalization
                    spec[..., :duration] = real

                    spec_x.append(spec)
                    dur_x.append(duration)
                
                tensors[use][lang]['spec'].append(torch.stack(spec_x, dim=0))
                tensors[use][lang]['length'].append(torch.stack(dur_x, dim=0))

            print(f'Lenght of: {use}, {lang} is {len(tensors[use][lang]["spec"])}')

    return tensors


def calc_mean(tensor_list):
    """
    Returns the mean and standard deviation
    of each tensors
    """
    sum_ = 0.0
    sum_sq = 0.0
    count = 0

    for tensor in tensor_list:
        for spec, duration in zip(tensor['spec'], tensor['length']):
            file = spec[..., :duration]

            sum_ += file.sum()
            sum_sq += (file**2).sum()
            count += 1

    mean = sum_ / count
    std = (sum_sq / count - mean**2).sqrt()

    return float(mean), float(std)


def print_data_mean(tensors):
    """
    Prints mean for tensors to ensure data
    is normalized

    Tensor: {use : {lang : {spect : []
                        {length : [] }}}}
    """

    def calc_sub_mean(data):
        """
        Calculates mean and std. on data organized
        like: 

        """

        sum_ = 0.0
        sum_sq = 0.0
        count = 0

        for spec_array, length_array in zip(data['spec'], data['length']):
            for spec, length in zip(spec_array, length_array):
                file = spec[..., :length]

                sum_ += file.sum()
                sum_sq += (file**2).sum()
                count += 1

        mean = sum_ / count
        std = (sum_sq / count - mean**2).sqrt()

        return float(mean), float(std)



    for use, dataset in tensors.items():
        mean = 0.0
        std = 0.0

        for _, data in dataset.items():

            mean_, std_ = calc_sub_mean(data)

            mean += mean_
            std += std_

        print(f"{use} has mean: {mean / len(dataset)}, std: {std / len(dataset)}")


def flatten_data(tensors, encoder):
    """
    Takes in the tensors dataset - now normalized with
    dict hiercachy and turns it into datasets
    """
    result = {"train": None, "test": None, "validation": None}

    for dtype, dataset in tensors.items():

        data_sets = []

        for lang, data in dataset.items():

            # Calculates the number of labels to assign
            values = len(data['spec']) * data['spec'][0].shape[0]

            x = list(data)
            y = encoder.transform([lang] * values)

            # Todo <-- Define new Dataset

            data_sets.append(
                TensorDataset(torch.cat(x, dim=0), torch.tensor(y, dtype=torch.long))
            )

        result[dtype] = data_sets

    return result


def build_loaders(tensors):
    """
    Creates out dataloaders for training
    """

    def load_data(data, shuffle):
        return DataLoader(
            data, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=shuffle
        )

    train = ConcatDataset(tensors["train"])
    test = ConcatDataset(tensors["test"])
    validation = ConcatDataset(tensors["val"])

    train_loader = load_data(train, True)
    test_loader = load_data(test, False)
    validation_loader = load_data(validation, False)

    return train_loader, test_loader, validation_loader


def load_tensors(languages, data_location, enc):
    """
    Loads the tensor data_loaders
    """

    tensors, reference = load_audio(languages, data_location)
    tensors = standardize_tensors(tensors)


    print_data_mean(tensors)

    train, test, val = build_loaders(flatten_data(tensors, enc))

    shape = reference['train']['en'].shape

    return train, test, val, shape
