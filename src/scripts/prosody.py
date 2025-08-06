from language_detection.spect import main as main_spect
from language_detection.train import main as main_train
from language_detection.data.spectrogram.functions import compute_lowpass_spectrogram_batch
from language_detection.model.network import VarCNNTransformerLanguageDetector



if __name__ == '__main__':
    # languages = ["en", "es", "de", "it", "ta", "nl", "ja"]
    languages = ["en", "de", "nl", "es", "it", "ja", "ta"]

    origin = "/om2/user/moshepol/prosody/data/raw_audio/"
    save = "/om2/user/moshepol/prosody/models/test/no_prosody_seven/"

    entry = {
        "sr" : 16_000,
        "n_fft" : 1024,
        "hop_length" : 512,
        "spect_f" : compute_lowpass_spectrogram_batch,
    }

    # Make Spectrogram
    main_spect(languages, entry, origin)

    # Train Model
    main_train(languages, VarCNNTransformerLanguageDetector, origin, save)