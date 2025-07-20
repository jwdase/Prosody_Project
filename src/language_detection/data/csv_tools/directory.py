import pandas as pd
from pathlib import Path

def open_files(language, size = None):
    '''
    Opens file that contains all the 
    paths to files, also includes speaker id
    '''

    root = '/om2/user/moshepol/prosody/data/raw_audio'
     
    path = (f'{root}/{language}/validated.tsv')
    dataframe = pd.read_csv(path, sep = '\t', low_memory = False)[['client_id', 'path']]

    print(f'Loading {language} from: {path}')

    if size is not None:
        return dataframe.sample(n = min(size, len(dataframe)))

    return dataframe


def person_to_group(df):
    '''
    Develops a dictionary that maps
    speaker id to duration
    '''
    return dict(tuple(df.groupby('client_id', sort=False)))



def make_single_string(name, maps):
    '''
    Takes a dictionary of lengths to URL's and joins URLs with hypthens
    Returns a dictionary suitable for making a pandas dataframe
    '''
    dictionary = {'id' : name}

    for val, urls in maps.items():
        dictionary[val] = [urls]

    return dictionary


def make_into_list(name, maps):
    '''
    Takes a dictionary of lengths to URL's and joins URLs with hypthens
    Returns a list that maps audion length correctly for insertion
    into pandas dataframe
    '''
    insert = [name]

    for _, urls in maps.items():
        insert.append(urls)

    return insert

def form_dataframe(name, values):
    """
    Turns values into a pandas Dataframe
    that we can then add more lines to
    """

    x = make_single_string(name, values)
    return pd.DataFrame(x)