import pickle

from language_detection.data.spectrogram.loader import create_dataset

def generate_dataset(lang):
    """
    Generates pickle files for our dataset which
    we'll use on the matlab file
    """
    
    x, y, total = create_dataset(lang)

    with open('src/notebooks/play/files.pkl', 'wb') as f:
        pickle.dump(x, f)

    with open('src/notebooks/play/speak.pkl', 'wb') as f:
        pickle.dump(y, f)

    with open('src/notebooks/play/shape.pkl', 'wb') as f:
        pickle.dump(total, f)

if __name__ == "__main__":
    lang = ["ta", "en", "es", "ja", "it", "de", "nl"]

    generate_dataset (lang)


