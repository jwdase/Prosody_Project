import torch
import numpy as np
from torch.utils.data import random_split

def build_tensor(data):
    """
    Creates a 2D np array, then adds another row
    to specifify 1 data entry then converts to
    pytorch tensor and normalizes
    """
    np_array = np.array(data, dtype = np.float32)
    np_array = np_array[:, np.newaxis, :, :]
    tensor = torch.tensor(np_array, dtype=torch.float32)

    # Normalize
    return (tensor + 80) / 80

def split_dataset(df, train):
    '''
    Takes in a tensor mapped from input to output
    and retursn
    '''
    train_size = int(len(df) * train)
    test_size = len(df) - train_size
    return random_split(df, [train_size, test_size])
