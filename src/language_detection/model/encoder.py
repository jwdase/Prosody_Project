from sklearn.preprocessing import LabelEncoder
import joblib

def generate_encoder(lang_list):
    """
    Creates the encoder for the respective languages
    """

    encoder = LabelEncoder()
    encoder.fit(list(lang_list))

    print(f"Classes: {encoder.classes_}")

    return encoder

def save_encoder(enc, loc):
    joblib.dump(enc, f"{loc}/label_encoder.pkl")