"""
Script to run a prosody based NN
"""

from path import setup_project_root
setup_project_root()

from low_pass_spect import main as make_spect
from train_gpu import main as run_train


if __name__ == '__main__':
    languages = ["en", "es", "de", "it"]
    window = "5.5 - 6.0"
    window_model = 'range_5_5-6_0'

    new_location = "/om2/user/moshepol/prosody/data/low_pass/"
    save_location = "/om2/user/moshepol/prosody/low_models/initial_model"

    num_epochs = 25
    samples_per_person = 10

    n_ftt = 2048
    hop_length = 512
    sr = 16_000

    entry = {"sr": sr, "n_fft": n_ftt, "hop_length": hop_length}

    make_spect(languages, window, samples_per_person, entry, new_location[:-1])
    run_train(languages, window_model, num_epochs, new_location, save_location)

