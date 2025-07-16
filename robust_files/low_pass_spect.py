"""
This script applies a low pass filter and thne uses it on 
the GPU
"""

import random

import torch
import torchaudio
import glob

from path import setup_project_root
setup_project_root()

from make_spect import main as spect_computation
from make_spect import check_path

AUDIO_SAVED = '/om2/user/moshepol/prosody/data/low_pass'
CUTOFF = 300
SR = 16_000

def compute_lowpass_spectrogram_batch(lang, batch, window, n_fft, hop_length):
    """
    Applies a low pass filer to a batch of audio
    and randomly saves 10% of audio files for reference later
    it then turns the audio into spectrograms
    """
    batch = batch.to('cuda')

    # Filter Audio
    filtered_audio, mask = fft_lowpass_batch(batch, SR, CUTOFF)

    # Saves some audio files
    choices(lang, filtered_audio, 2, AUDIO_SAVED)

    # Retursn complex
    specs = torch.stft(
        filtered_audio.squeeze(1),
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
        window=window,
    )

    power = specs.abs() ** 2
    db = 10 * torch.log10(torch.clamp(power, min=1e-10))

    # Removes the frequency bins that are above our cutoff
    freqs = torch.fft.rfftfreq(n_fft, 1 / sr).to('cuda')
    mask = freqs <= CUTOFF
    db_trimmed = db[:, mask, :]

    return db_trimmed.cpu()

def fft_lowpass_batch(audio, sr, cutoff):
    '''
    Applies a sharp FFT based low-pass filter to audio

    Args:
        audio : Tensor of shape [B, 1, T]
        sr: audio sample rate 16,000
        cutoff: where we want the low pass cutoff to end

    '''

    audio = audio.squeeze(1) # [B, 1, T] --> [B, T]

    # FFT
    fft = torch.fft.rfft(audio, dim=1)
    freqs = torch.fft.rfftfreq(audio.shape[-1], d=1/sr).to('cuda')

    # Creates a mask and applies it
    mask = freqs <= cutoff
    fft = fft * mask

    # Turn back into audio
    audio = torch.fft.irfft(fft, n=audio.shape[-1], dim=-1).unsqueeze(1) # [B, 1, T]
    return audio, mask



def choices(lang, audio, num_samples, directory):
    """
    Saves the audio file under the directory specified
    """
    directory = f'{AUDIO_SAVED}/{lang}/recordings'

    saved_files = random.sample(range(audio.size(0)), num_samples)

    start = last_audio(directory)

    for name, i in enumerate(saved_files):
        path = f'{directory}/{make_name(start + name + 1)}'
        torchaudio.save(path, audio[i].cpu(), SR)

    return None


def make_name(name):
    '''
    Creates file name
    '''
    pad = 4 - len(str(name))
    return pad * '0' + str(name) + '.wav'


def last_audio(path):
    '''
    Figures out the numerical value of the last
    file_name saved
    '''

    try:
        return int(glob.glob(path + '/*.wav')[-1].split('/')[-1].replace('.wav', '').lstrip('0'))
    except IndexError:
        return 0

def main(languages, time_frame, num_speakers, audio_process, new_location, spect_func):
    """
    Runs the complete make_spect code
    """

    # Empty directory before making spectrograms
    for lang in languages:
        path = f'{AUDIO_SAVED}/{lang}/recordings/'
        check_path(path)
        print(f'lang: {lang} recordings are saved: {path}')

    # Make the spectrograms
    spect_computation(languages, time_frame, num_speakers, audio_process, new_location, spect_func)

if __name__ == '__main__':

    languages = ["en", "es", "de", "it"]
    window = "5.5 - 6.0"

    location = "/om2/user/moshepol/prosody/data/low_pass"

    n_ftt = 2048
    hop_length = 512
    sr = 16_000

    entry = {"sr": sr, "n_fft": n_ftt, "hop_length": hop_length}

    main(languages, window, 1, entry, location, compute_lowpass_spectrogram_batch)



