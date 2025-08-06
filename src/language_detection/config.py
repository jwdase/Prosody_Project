import torch

# Device
DEVICE = "cuda"

# Data Loaders
BATCH_SIZE = 256
NUM_WORKERS = 4

# Training
NUM_EPOCHS = 25
LR = 1e-4

# Scheduler
PATIENCE = 2
FACTOR = .5
THRESHOLD = 1e-4
ERROR = .005

# Number of Unique Speakers
NUM_SPEAKERS = 25
SPECT_SIZE = 128
MAX_LENGTH = 9.5
WORKERS = 1

# Used for making spectrograms
WINDOW = torch.hann_window

# Audio Location
AUDIO_LOCATION = '/om2/user/moshepol/prosody/data/raw_audio'

# Low Pass Filter, Location to save Audio
CUTOFF = 300
AUDIO_SAVED = "/om2/user/moshepol/prosody/data/low_pass"

# Weights for Audio File Length
WEIGHTS = {
    "0.0 - 0.5": 0,
    "0.5 - 1.0": 0,
    "1.0 - 1.5": 0,
    "1.5 - 2.0": 0,
    "2.0 - 2.5": 0,
    "2.5 - 3.0": 10,
    "3.0 - 3.5": 10,
    "3.5 - 4.0": 10,
    "4.0 - 4.5": 10,
    "4.5 - 5.0": 10,
    "5.0 - 5.5": 5,
    "5.5 - 6.0": 5,
    "6.0 - 6.5": 5,
    "6.5 - 7.0": 5,
    "7.0 - 7.5": 5,
    "7.5 - 8.0": 5,
    "8.0 - 8.5": 5,
    "8.5 - 9.0": 5,
    "9.0 - 9.5": 10,
    "9.5 - 10.0": 0,
}