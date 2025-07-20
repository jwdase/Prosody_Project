import torch

# Device
DEVICE = "cuda"

# Data Loaders
BATCH_SIZE = 128
NUM_WORKERS = 4

# Training
NUM_EPOCHS = 25
LR = 1e-4

# Scheduler
PATIENCE = 2
FACTOR = .5
THRESHOLD = 1e-4

# Number of Unique Speakers
NUM_SPEAKERS = 4

# Used for making spectrograms
WINDOW = torch.hann_window

# Low Pass Filter, Location to save Audio
CUTOFF = 300
AUDIO_SAVED = "/om2/user/moshepol/prosody/data/low_pass"