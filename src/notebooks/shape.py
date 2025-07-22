"""
Script to figure out our dataset distribution and then load the
audio files for creating our spectrograms
"""

from language_detection.data.csv_tools.read import (
    load_df,
    key_length,
    convert_to_list,
    num_speakers,
)

from language_detection import config

import random


def find_sizes(languages):
    """
    Works to find the maximum size of each
    column for our dataset as inputed
    """
    dataset = load_df(languages)

    time_size = {val: 100_000 for val in dataset[languages[0]].columns[2:]}

    for lang, df in dataset.items():
        for key, value in time_size.items():
            time_size[key] = min(time_size[key], key_length(df, key))

    return time_size


def smallest_lang(languages):
    """
    Returns the smalles languages by
    unique speakers in the list given
    """
    return sorted(num_speakers(languages).items(), key=lambda x: x[1])[0][0]


def fill_smallest(lang, maximum):
    """
    Begins the filling process on the smallest
    language
    """

    df = load_df([lang])[lang]
    current = {key: 0 for key, _ in maximum.items()}

    audio_files = []

    i = 0
    for index, row in list(df.iterrows()):
        d = make_dict(row)
        audio_files.extend(first_add(d, maximum, current))
        i += 1
    
    return audio_files, current


def make_dict(row):
    """
    Turns each row into a dictionary with a list
    of options mapped to duration
    """
    x = {}

    for title in config.WEIGHTS.keys():
        x[title] = convert_to_list(row[title])

    return x


def first_add(options, maximum, current):
    """
    Returns which values should be added first
    as a list - uses a function to determine order
    """

    opt = {
        key: sources 
        for key, sources in options.items() 
        if len(sources) != 0
    }

    total = sum([len(sources) for _, sources in opt.items()])

    # Speaker has spoken less than upper limit
    # so we return all of the speakers files
    if total <= config.NUM_SPEAKERS:
        return insert(opt, maximum, current)

    # If the speaker has more files, we need
    # to add only a select amount

    return select_audio(opt, maximum, current)


def select_audio(options, maximum, current):
    """
    Weighs audio file to have more longer clips
    but fills the dataset with the maximum amount of
    audio files
    """

    def probability_f():
        """
        Returns how we want to weigh probabilities in 
        data selection, ensures weights always
        add up to 1
        """

        weights = [config.WEIGHTS[key] for key in options.keys()]
        multiple = 1 / sum(weights)

        return [multiple * val for val in weights]

    weights = probability_f()

    chosen_files = set()

    while len(chosen_files) < config.NUM_SPEAKERS:
        frame, files = random.choices(list(options.items()), k=1, weights=weights)[0]
        file = random.choice(files)

        # Ensures we can add file and not repeating
        if (not check_less(file, frame, maximum, current) 
            or file in chosen_files):
            continue

        # Adds file to x if valid
        current[frame] += 1
        chosen_files.add(file)

    return list(chosen_files)


def check_less(val, frame, maximum, current):
    """
    Checks to make sure we can add our audio
    recording without surpassing limits 
    """
    return maximum[frame] > current[frame] + len(val)


def insert(options, maximum, current):
    """
    Returns only the values that don't exceed the
    maximum total size
    """
    x = []

    for frame, val in options.items():
        if check_less(val, frame, maximum, current):
            current[frame] += len(val)
            x += val

    return x


def fill(lang, size):
    pass

def priority_function(options, size, current):
    """
    Tell's the code which bucket to add each audio file
    to untill all buckets are filled
    """



def create_dataset(languages):
    """
    Runs the majority of the code to create a 
    """

    # Audio files will all be saved key : val
    dataset = {}

    maximum = find_sizes(languages)
    smallest_language = smallest_lang(lang)

    audio_files, maximum = fill_smallest(smallest_language, maximum)
    dataset[smallest_language] = audio_files

    for lang in languages:
        if lang is smallest_language:
            continue











if __name__ == "__main__":
    lang = ["ta", "en", "es"]
    # print(smallest_lang(lang))
    maximum = {
        "0.0 - 0.5": 0,
        "0.5 - 1.0": 0,
        "1.0 - 1.5": 0,
        "1.5 - 2.0": 6,
        "2.0 - 2.5": 134,
        "2.5 - 3.0": 647,
        "3.0 - 3.5": 2048,
        "3.5 - 4.0": 3266,
        "4.0 - 4.5": 4245,
        "4.5 - 5.0": 4351,
        "5.0 - 5.5": 4070,
        "5.5 - 6.0": 3713,
        "6.0 - 6.5": 3318,
        "6.5 - 7.0": 2708,
        "7.0 - 7.5": 2452,
        "7.5 - 8.0": 2263,
        "8.0 - 8.5": 1868,
        "8.5 - 9.0": 1817,
        "9.0 - 9.5": 3005,
        "9.5 - 10.0": 0,
    }
    print(maximum)

    fill_smallest("ta", maximum)
