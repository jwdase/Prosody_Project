from language_detection.train import main
from language_detection.model.network import VarCNNTransformerLanguageDetector



if __name__ == "__main__":
    language = ["ja", "uz",]

    main(language, VarCNNTransformerLanguageDetector, name='Model1')