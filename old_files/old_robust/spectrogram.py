import librosa
import numpy as np

def make_spect(file):
    '''
    Takes in a file and creates the spectrogram
    '''
    audio, sr = librosa.load(file, sr = 22)

    hop_length = 512

    D = librosa.stft(audio, hop_length=hop_length)
    return librosa.amplitude_to_db(abs(D), ref = np.max)



def extendo(audio, shape):
    '''
    Takes in a spectrogram and lengthens
    it to ensure that it is all of size
    (1025, 216)
    '''

    freq, length = audio.shape

    if length < shape:
        pad_amount = shape - length
        return np.pad(audio, pad_width=((0, 0), (0, pad_amount)), mode='constant', constant_values = -80)

    if length != shape:
        print(audio.shape)
        raise TypeError

    return audio    

def parse(files, shape = 216):
    '''
    Creates a spectrogram for all of our audio files
    '''
    spectrograms = []

    for file in files:
        spect = make_spect(file)
        spectrograms.append(extendo(spect, shape))

    return spectrograms    