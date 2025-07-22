import random
import pandas as pd

from language_detection import config

def convert_to_list(val):
    '''
    Takes in the string output from the .csv and 
    turns it into a list 
    '''
    x = []

    # Empty Value
    if val == '[]':
        return []

    # Only one entry
    if ',' not in val:
        return [val.replace("[", '').replace("]", '').replace("'", '').replace(' ', '')]

    # Otherwise Multiple entries
    array = []
    for x in val.split(','):
        array.extend(convert_to_list(x))

    return array


def select_array(df, key, num_speaker_record):
    '''
    Returns all links that work for a certain 
    language with-in that timeframe
    '''
    
    test_array = []
    validation_array = []
    train_array = []

    for i, x in enumerate(df[key]):

        try:
            val = convert_to_list(x)[0:num_speaker_record]
        except IndexError:
            val = convert_to_list(x)

        random_var = random.random() * 100

        if 0 < random_var < 10:
            test_array.extend(val)
        elif 30 < random_var < 40:
            validation_array.extend(val)
        else:
            train_array.extend(val)

    return {'test' : test_array, 'val' : validation_array, 'train' : train_array}


def key_length(df, key):
    '''
    Takes in a df and a certain key
    and adds up all the values of that column
    '''

    return sum([len(x) for _, x in select_array(df, key, 100).items()])

def load_df(languages):
    """
    loads custom .csv from each respective language
    """

    lang_to_df = {}

    for lang in languages:
        lang_to_df[lang] = pd.read_csv(
            f'{config.AUDIO_LOCATION}/{lang}/custom/length.csv'
        )
        print(
            f'Loaded: {lang} from {config.AUDIO_LOCATION}/{lang}/custom/length.csv'
            )
    
    return lang_to_df


def num_speakers(languages):
    """
    Calculates the number of unique speakers
    per a column
    """

    lang_to_df = load_df(languages).items()

    return {lang : df.shape[0] for lang, df in lang_to_df}