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

    train_files = {key: [] for key, _ in maximum.items()}
    test_file = {key: [] for key, _ in maximum.items()}
    val_file = {key: [] for key, _ in maximum.items()}

    for index, row in list(df.iterrows()):
        opt = make_dict(row)

        if opt_size(opt) > config.NUM_SPEAKERS:
            opt = select_audio(
                opt, maximum, current_size(train_files, test_file, val_file)
            )

        # Logic for choosing which dict to add
        x = random.randint(0, 100)
        if x <= 15:
            add_dict(test_file, opt)
        elif 15 <= x <= 20:
            add_dict(train_files, opt)
        else:
            add_dict(val_file, opt)

    return train_files, test_file, val_file


def add_dict(main, new):
    """
    Adds new values to our larger dictionary of audio
    files
    """
    for key, value in new.items():
        main[key] += value


def current_size(d1, d2, d3):
    return {
        key: len(i1) + len(i2) + len(i3)
        for (key, i1), (_, i2), (_, i3) in zip(d1.items(), d2.items(), d3.items())
    }


def opt_size(dictionary):
    """
    figures out how many audio files this
    person has generated
    """
    return sum([len(val) for val in dictionary.values()])


def make_dict(row):
    """
    Turns each row into a dictionary with a list
    of options mapped to duration
    """
    x = {}

    for title in config.WEIGHTS.keys():
        if convert_to_list(row[title]) != []:
            x[title] = convert_to_list(row[title])
    return x


def select_audio(options, maximum, current):
    """
    Weighs audio file to have more longer clips
    but fills the dataset with the maximum amount of
    audio files
    """

    def probability_f(opt):
        """
        Returns how we want to weigh probabilities in
        data selection, ensures weights always
        add up to 1
        """

        weights = [config.WEIGHTS[key] for key in opt.keys()]
        multiple = 1 / sum(weights)

        return [multiple * val for val in weights]

    def add_choosen(dictionary, key, val):
        """
        Adds values to the dictionary based on it's key and
        value
        """
        if key in dictionary:
            dictionary[key].append(val)

        else:
            dictionary[key] = [val]


    def remove_opt(dictionary, key, file):
        """
        Removes the link, if no longer a link in
        datastructure removes that key as well
        """
        dictionary[key].remove(file)

        if len(dictionary[key]) == 0:
            del dictionary[key]


    weights = probability_f(options)
    chosen_files = {}

    i = 0
    j = 0

    while j < config.NUM_SPEAKERS and i < config.NUM_SPEAKERS * 3:
        frame, files = random.choices(list(options.items()), k=1, weights=weights)[0]
        file = random.choice(files)

        # Removes from the dictionary
        remove_opt(options, frame, file)

        # Updates weights
        weights = probability_f(options)

        # Ensures we can add file and not repeating
        if not check_less(file, frame, maximum, current):
            i += 1
            continue

        # Adds file to x if valid
        add_choosen(chosen_files, frame, file)
        j += 1

    return chosen_files


def check_less(val, frame, maximum, current):
    """
    Checks to make sure we can add our audio
    recording without surpassing limits
    """
    return maximum[frame] > current[frame] + len(val)

#############-----------------------################

def fill(lang, size):
    pass


def priority_function(options, size, current):
    """
    Tell's the code which bucket to add each audio file
    to untill all buckets are filled
    """
    pass


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
    # print(maximum)

    d1, d2, d3 = fill_smallest("ta", maximum)

    total = 0

    for key, value in current_size(d1, d2, d3).items():
        print(key, value)
        total += value

    # for key, value in files.items():
    #     print(key, value)

    print(f"Total Audio Recordings: {total}")
