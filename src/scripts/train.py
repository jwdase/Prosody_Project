from language_detection.train import main
from language_detection.model.network import VarCNNTransformerLanguageDetector



if __name__ == "__main__":
    language = ["en", "de", "nl", "es", "it", "ja", "ta",]

    origin = '/om2/user/jwdase/prosody/prosody_only_spect/'
    base = '/om2/user/jwdase/prosody/models/test/prosody_epoch_20_tr/'

    main(language, VarCNNTransformerLanguageDetector, origin, base)