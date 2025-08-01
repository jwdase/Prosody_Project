from torch.utils.data import Dataset
import torchaudio
import torch

class AudioFileDataset(Dataset):
    """
    We have to use a Dataset to load the Audio signal
    onto the GPU, then the Spectrogram computation occurs
    on the GPU
    """

    def __init__(self, audio_dir, sr, max_time):
        self.files = list(audio_dir)
        self.target_sr = sr
        self.length = int(max_time * sr)
        self.samplers = {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        waveform, sr = torchaudio.load(path)
        original_length = waveform.shape[-1]

        if sr != self.target_sr:

            if sr not in self.samplers:
                self.samplers[sr] = torchaudio.transforms.Resample(sr, self.target_sr)

            waveform = self.samplers[sr](waveform)

        return self.pad_waveform(waveform), original_length

    def pad_waveform(self, waveform):
        length = waveform.shape[-1]

        if length > self.length:
            print('Clipped')
            waveform = waveform[..., :self.length]
            length = self.length 
            
        pad = self.length - length
        return torch.nn.functional.pad(waveform, (0, pad))