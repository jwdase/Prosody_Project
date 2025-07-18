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