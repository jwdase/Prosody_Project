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

def run_spect(languages, time_frame, num_speakers, audio_home, new_location):
    from make_spect import main as spect_main
    spect_main(languages, time_frame, num_speakers, audio_home, new_location)
    del spect_main

def run_train(languages, time_frame, num_epochs, data_location, new_location):
    from train_gpu import main as train_main
    train_main(languages, time_frame, num_epochs, data_location, new_location)
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

    new_location = "/om2/user/moshepol/prosody/data/low_pass/"
    save_location = "/om2/user/moshepol/prosody/models/"

    num_epochs = 2
    samples_per_person = 1

    for parameters in create_options((sr, n_fft, hop_length)):
        param = {'sr' : parameters[0], 'n_fft' : parameters[1], 'hop_length' : parameters[2]}

        # Builds the spectrogram according to new parameters
        run_spect(languages, window_spect, samples_per_person, param, new_location)

        base = create_base(param)
        
        # Trains the spectrogram then saves it
        run_train(languages, window_model, num_epochs, new_location, save_location)

        # Empties GPU
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Empties RAM
        gc.collect()

        print(torch.cuda.memory_summary())
        print(f'old_parameters: {param}')
        print(f"CPU RAM used: {psutil.virtual_memory().used / 1e9:.2f} GB")





