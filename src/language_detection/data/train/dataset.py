from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self):
        self.spects = []
        self.lengths = []
        self.labels = []

    def __len__(self):
        return len(self.spects)

    def addfiles(self, specs, lengths, labels):
        assert len(specs) == len(lengths) == len(labels), "Mismatched lengths"

        self.spects += specs
        self.lengths += lengths
        self.labels += labels

    def __getitem__(self, idx):
        return self.spects[idx], self.lengths[idx], self.labels[idx]

