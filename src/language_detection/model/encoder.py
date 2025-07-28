import numpy as np

from sklearn.preprocessing import LabelEncoder
import joblib

def generate_encoder(lang_list):
    """
    Creates the encoder for the respective languages
    """

    encoder = LabelEncoder()
    encoder.fit(list(lang_list))

    x = {val : i for i, val in enumerate(encoder.classes_)}

    print(f"Classes: {x}")

    return encoder

def save_encoder(enc, loc):
    print('enter')
    joblib.dump(enc, f"{loc}/label_encoder.pkl")

class CustomLabelEncoder:
    def __init__(self, class_order):
        self.class_to_index = {label: idx for idx, label in enumerate(class_order)}
        self.index_to_class = {idx: label for label, idx in self.class_to_index.items()}
        self.classes_ = np.array(class_order)

    def transform(self, labels):
        return np.array([self.class_to_index[label] for label in labels])

    def inverse_transform(self, indices):
        return np.array([self.index_to_class[idx] for idx in indices])