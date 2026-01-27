from mutagen import File
from mutagen.mp3 import HeaderNotFoundError, MutagenError
import random
from pathlib import Path

from language_detection import config

def get_length(path):
    '''
    Attemps to get length of audio
    if path is not there, ignores
    '''
    
    try:
        audio = File(path)
        return audio.info.length
    except MutagenError as e:
        # Signal path is not valid
        return None


    
def language_path_builder(df, language):
    '''
    Because some language's have different path's
    for their respective url this function returns the
    correct one
    '''

    path = f'{config.AUDIO_LOCATION}/{language}/clips/'

    ending = '.wav'

    return (path + file.replace('.mp3', '.wav') for file in df['path'])


def valid_paths(df, length, delta, language):
    '''
    return the options of paths that will work
    given the size of the audio file
    '''

    paths = language_path_builder(df, language)

    options = []

    count = 0

    for path in paths:
        file_len = get_length(path)

        if file_len is None:
            count += 1
            continue

        if length - delta <= file_len <= length:
            options.append(path)
    
    return options


def df_values(df, language):
    '''
    Takes in an df that contains all the common voice audio
    files for a repective speaker, and returns them segmented
    into a dictionary

    dict_key : length
    dict_item : list of urls
    '''

    # bins we will sort it into, cannot refactor it
    BIN_NUM = 20
    BINS = [val / 2 for val in range(BIN_NUM)]

    bins = {f'{val} - {val + .5}'  : [] for val in BINS}
    keys = list(bins.keys())

    for url in language_path_builder(df, language):
        length = get_length(url)

        if length is None:
            continue
        
        # inserts it into correct index
        i = 0
        while i < BIN_NUM - 2 and length > BINS[i]:
            i += 1

        # inserts value
        bins[keys[i]].append(url)

    return bins



def random_audio(files, choices):
    '''
    Selects the number of audio files fron each speaker
    and adds them to a list
    '''
    return random.sample(files, min(choices, len(files)))



 
