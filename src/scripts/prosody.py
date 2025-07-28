from language_detection.spect import main as main_spect
from language_detection.train import main as main_train
from language_detection.data.spectrogram.functions import compute_lowpass_spectrogram_batch
from language_detection.model.network import CNNLanguageDetector, CNNRNNLanguageDetector



if __name__ == '__main__':
    languages = ["en", "de", "nl", "es", "it", "ja", "ta"]
    window = "4.0 - 4.5"
    acess_window = "range_4_0-4_5"

    origin = "/om2/user/moshepol/prosody/data/low_pass/"
    save = "/om2/user/moshepol/prosody/models/test/four_lang_prosody/"

    entry = {
        "sr" : 16_000,
        "n_fft" : 1024,
        "hop_length" : 512,
        "spect_f" : compute_lowpass_spectrogram_batch,
        "length" : 4.5
    }

    # Make Spectrogram
    main_spect(languages, window, entry, origin)
    
    # Train Model
    main_train(languages, acess_window, CNNRNNLanguageDetector, origin, save)
