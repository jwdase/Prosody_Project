"""
This Script makes a language detection model, it is trained
on spectrograms which must be first created in the make_spect file
"""

from language_detection.utils.io import grab_device, check_path, save_test, save_model
from language_detection.model.encoder import generate_encoder, save_encoder, CustomLabelEncoder
from language_detection.model.loader import load_tensors
from language_detection.model.network import VarCNNRNNLanguageDetector, VarCNNRNNLanguageDetector2, VarCNNTransformerLanguageDetector
from language_detection.model.train import train_loop
from language_detection.model.evaluate import plot_loss, plot_lr
from language_detection import config

def main(languages, mod, name):
    """
    Main training loop
    """

    # Get locations for data and save
    model_location = f"{config.MODEL_LOCATION}/{name}"
    check_path(model_location)


    # Checks using CUDA and clears directory to save files
    grab_device()

    # Load tensors and encoders
    encoder = CustomLabelEncoder(languages)
    train, test, val, shape = load_tensors(
        languages,
        encoder
    )

    # Create the model
    model = mod(len(languages), shape)

    # Trains the Model
    total_loss, val_loss, lr_plot = train_loop(
        model,
        train,
        val,
        model_location
    )

    # Saving Model + Statistics on Training
    plot_loss(total_loss, val_loss, model_location)
    plot_lr(lr_plot, model_location)
    save_encoder(encoder, model_location)
    save_test(test, model_location)

    print(f'Data saved to: {model_location}')

if __name__ == '__main__':
    language = ["en", "de", "nl", "es", "it", "ja", "ta",]

    main(language, VarCNNTransformerLanguageDetector, name="None")