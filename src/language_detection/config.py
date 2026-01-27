import torch

# Device
DEVICE = "cuda"

# Data Loaders
BATCH_SIZE = 212
NUM_WORKERS = 4

# Training
NUM_EPOCHS = 25
LR = 1e-4

# Scheduler
PATIENCE = 2
FACTOR = .5
THRESHOLD = 1e-4

# Saving model
ERROR = .005

# Number of Unique Speakers
NUM_SPEAKERS = 25
SPECT_SIZE = 212
MAX_LENGTH = 9.5
WORKERS = 8

# Used for making spectrograms
WINDOW = torch.hann_window

# Audio Location
AUDIO_LOCATION = '/orcd/archive/evelina9/001/u/jwdase/Augmented_Data'
MODEL_LOCATION = '/orcd/archive/evelina9/001/u/jwdase/Model'

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
    "3.0 - 3.5": 1,
    "3.5 - 4.0": 1,
    "4.0 - 4.5": 1,
    "4.5 - 5.0": 1,
    "5.0 - 5.5": 1,
    "5.5 - 6.0": 1,
    "6.0 - 6.5": 1,
    "6.5 - 7.0": 1,
    "7.0 - 7.5": 1,
    "7.5 - 8.0": 1,
    "8.0 - 8.5": 1,
    "8.5 - 9.0": 1,
    "9.0 - 9.5": 1,
    "9.5 - 10.0": 1,
}