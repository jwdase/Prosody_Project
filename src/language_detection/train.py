"""
This Script makes a language detection model, it is trained
on spectrograms which must be first created in the make_spect file
"""

from language_detection.utils.io import grab_device, check_path, save_test, save_model
from language_detection.model.encoder import generate_encoder, save_encoder
from language_detection.data.loader import load_tensors
from language_detection.model.network import CNNLanguageDetector
from language_detection.model.train import train_loop
from language_detection.model.evaluate import plot_loss

def main(languages, time_frame, mod, data_location, new_location):
    """
    Main training loop
    """

    # Checks using CUDA and clears directory to save files
    grab_device()
    check_path(new_location)

    # Load tensors and encoders
    encoder = generate_encoder(languages)
    train, test, val, shape = load_tensors(
        languages,
        time_frame,
        data_location,
        encoder
    )

    # Create the model
    model = mod(len(languages), shape)

    # Trains the Model
    total_loss, val_loss = train_loop(
        model,
        train,
        val,
        new_location
    )

    # Saving Model + Statistics on Training
    plot_loss(total_loss, val_loss, new_location)
    save_encoder(encoder, new_location)
    save_test(test, new_location)

    print('Done')

if __name__ == '__main__':
    language = ["en", "es", "de"]
    window = "range_5_5-6_0"

    origin = '/om2/user/moshepol/prosody/data/raw_audio'
    base = '/om2/user/moshepol/prosody/models/test/'

    main(language, window, LanguageDetector, origin, base)