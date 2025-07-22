'''
This script generates a .tsv with speaker id as the row
and files of different lengths in each column
'''

from pathlib import Path
import pandas as pd

# Setup Project Root
from path import setup_project_root
setup_project_root()

# Module imports
from langauge_detection.data.com_voice_dir import open_files, person_to_group
from langauge_detection.data.audio_length import df_values


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

def main(language):
    '''
    Main run through of code
    '''
    print(f'Language: {language}')

    # Load Language Files
    files = person_to_group(open_files(language))

    # Constructing Dataframe
    name, person_data = files.popitem()

    # Creates the structure for pandas dataframe
    x = make_single_string(name, df_values(person_data, language))
    df = pd.DataFrame(x)

    # Loads the rest of the files
    index = 1
    for name, urls in files.items():
        df.loc[index] = make_into_list(name, df_values(urls, language))
        index += 1

    # Make folder and save it
    path = Path(f'/om2/user/moshepol/prosody/data/raw_audio/{language}/custom')
    path.mkdir(exist_ok=True)
    df.to_csv(f'/om2/user/moshepol/prosody/data/raw_audio/{language}/custom/length.csv')

    print('Done')


if __name__ == '__main__':
    main('it')
