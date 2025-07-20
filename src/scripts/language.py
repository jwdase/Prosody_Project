from language_detection.spect import main as main_spect
from language_detection.train import main as main_train
from language_detection.data.spectrogram.functions import compute_spectrogram_batch
from language_detection.model.network import CNNLanguageDetector, CNNRNNLanguageDetector

if __name__ == '__main__':
    languages = ["en", "es", "de", "it"]
    window = "5.5 - 6.0"
    acess_window = "range_5_5-6_0"

    origin = "/om2/user/moshepol/prosody/data/raw_audio/"
    save = "/om2/user/moshepol/prosody/models/test/"

    entry = {
        "sr" : 16_000,
        "n_fft" : 1024,
        "hop_length" : 512,
        "spect_f" : compute_spectrogram_batch,
        "length" : 6.0
    }

    # Make Spectrogram
    main_spect(languages, window, entry, origin)

    # Train Model
    main_train(languages, acess_window, CNNRNNLanguageDetector, origin, save)