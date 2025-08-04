import numpy as np
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self):
        self.spectrograms = []
        self.lengths = []
        self.labels = []

        self.standardized = False

    def __len__(self):
        return len(self.spectrograms)

    def additems(self, spectrogram, lengths, labels):
        assert len(spectrogram) == len(lengths) == len(labels)

        self.spectrograms.extend(spectrogram)
        self.lengths.extend(lengths)
        self.labels.extend(labels)

    def additem(self, spectrogram, lengths, labels):
        self.additems([spectrogram], [lengths], [labels])

    def __getitem__(self, idx):
        return self.spectrograms[idx], self.lengths[idx], self.labels[idx]

    def getstats(self):
        means = []
        stds = []

        for i in range(len(self.spectrograms)):
            file = self.spectrograms[i][..., self.lengths[i]]

            means.append(file.mean())
            stds.append(file.std())
        
        return np.mean(means), np.mean(stds)

    def applynormal(self, mean, std):
        for i in range(len(self.spectrograms)):
            self.spectrograms[i][..., self.lengths[i]] = (
                (self.spectrograms[i][..., self.lengths[i]] - mean) / std
            )

        return 

    def print_stats(self):
        mean, std = self.getstats()

        print(f'Mean is: {mean}')
        print(f'Std is: {std}')

    def get_shape(self):
        return self.spectrograms[0].shape

