"""
This script tests a variety of different
ML models based on spectrogram type - we test
n_fft 
sr
hop_length 
"""

import torch
import psutil
import gc

from path import setup_project_root
setup_project_root()

def run_spect(languages, window_spect, param):
    from make_spect import main as spect_main
    spect_main(languages, window_spect, param)
    del spect_main

def run_train(languages, window_model, num_epochs, base):
    from Multi_Lang_GPU import main as train_main
    train_main(languages, window_model, num_epochs, base)
    del train_main


def create_options(choices):
    '''
    Recusrsive code to work through all the options
    for n_fft
    '''
    if len(choices) == 1:
        return [(opt, ) for opt in choices[0]]

    x = []
    for val in choices[0]:
        x += [(val,) + result for result in create_options(choices[1:])]

    return x

def create_base(params):
    '''
    Builds the directory for this data to be stored
    subdirectory: /om2/user/moshepol/prosody/models/

    '''
    base = '/om2/user/moshepol/prosody/models/'

    for key, item in params.items():
        base += (key + '=' + str(item))

    return base 

if __name__ == '__main__':
    sr = [16_000]
    n_fft = [2048, 512, 256]
    hop_length = [512, 256]

    languages = ["en", "it", "es", "de"]
    window_spect = "5.5 - 6.0"
    window_model = 'range_5_5-6_0'

    num_epochs = 25

    for parameters in create_options((sr, n_fft, hop_length)):
        param = {'sr' : parameters[0], 'n_fft' : parameters[1], 'hop_length' : parameters[2]}

        # Builds the spectrogram according to new parameters
        run_spect(languages, window_spect, param)

        base = create_base(param)
        
        # Trains the spectrogram then saves it
        run_train(languages, window_model, num_epochs, base)

        # Empties GPU
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Empties RAM
        gc.collect()

        print(torch.cuda.memory_summary())
        print(f'old_parameters: {param}')
        print(f"CPU RAM used: {psutil.virtual_memory().used / 1e9:.2f} GB")





