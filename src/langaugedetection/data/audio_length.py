from mutagen import File
from mutagen.mp3 import HeaderNotFoundError
import random
from pathlib import Path

def get_length(path):
    '''
    Attemps to get length of audio
    if path is not there, ignores
    '''
    try:
        audio = File(path)
        return audio.info.length
    except HeaderNotFoundError as e:
        # Signal path is not valid
        return None 
    
def language_path_builder(df, language):
    '''
    Because some language's have different path's
    for their respective url this function returns the
    correct one
    '''

    assert language in {'en', 'it'}

    root = str(Path(__file__).resolve().parents[3])

    if language == 'en':
        return (root + '/data/' + language + '/clips/' + path + '.mp3' for path in df['path'])
    if language == 'it':
        return (root + '/data/' + language + '/clips/' + path for path in df['path'])
    

def valid_paths(df, length, delta, language):
    '''
    return the options of paths that will work
    given the size of the audio file
    '''

    paths = language_path_builder(df, language)

    options = []

    for path in paths:
        file_len = get_length(path)

        if file_len is None:
            continue

        if length - delta <= file_len <= length:
            options.append(path)
    
    return options


def random_audio(files, choices):
    '''
    Selects the number of audio files fron each speaker
    and adds them to a list
    '''
    return random.sample(files, min(choices, len(files)))



 
