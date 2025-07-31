import torch

# Device
DEVICE = "cpu"

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

# Number of Unique Speakers
NUM_SPEAKERS = 25
SPECT_SIZE = 32
MAX_LENGTH = 9.0
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
    "0.0 - 0.5": 1,
    "0.5 - 1.0": 1,
    "1.0 - 1.5": 1,
    "1.5 - 2.0": 1,
    "2.0 - 2.5": 1,
    "2.5 - 3.0": 1,
    "3.0 - 3.5": 5,
    "3.5 - 4.0": 10,
    "4.0 - 4.5": 10,
    "4.5 - 5.0": 5,
    "5.0 - 5.5": 5,
    "5.5 - 6.0": 5,
    "6.0 - 6.5": 5,
    "6.5 - 7.0": 5,
    "7.0 - 7.5": 5,
    "7.5 - 8.0": 5,
    "8.0 - 8.5": 5,
    "8.5 - 9.0": 5,
    "9.0 - 9.5": 5,
    "9.5 - 10.0": 0,
}