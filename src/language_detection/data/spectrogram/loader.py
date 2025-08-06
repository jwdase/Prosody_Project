"""
Script to figure out our dataset distribution and then load the
audio files for creating our spectrograms
"""

import pickle
import random

from language_detection.data.csv_tools.read import (
    load_df,
    key_length,
    convert_to_list,
    num_speakers,
)

from language_detection import config

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
    Fills the smallest function first
    """

    def probability_f(opt, _):
        """
        Returns how we want to weigh probabilities in
        data selection, ensures weights always
        add up to 1
        """

        weights = [config.WEIGHTS[key] for key in opt.keys()]
        multiple = 1 / sum(weights)

        return [multiple * val for val in weights]

    def insert_f(files, opt, file_count):
        """
        Defines how we insert the file into
        the dataset
        """
        x = random.choices(["train", "test", "val"], weights=[0.8, 0.15, 0.05], k=1)[0]
        add_dict(files[x], opt)

        file_count[x].append(opt_size(opt))

    return fill(lang, maximum, probability_f, insert_f)


def fill_other(lang, maximum, max_dtype):
    """
    Fills languages besides the first language
    """

    def probability_f(opt, current):
        """
        Returns the priority of different keys functionis
        1 - % full in priority
        """
        pecentages = []

        for key, _ in opt.items():
            try:
                pecentages.append(max(0, 1 - current[key] / maximum[key]))
            except ZeroDivisionError:
                pecentages.append(0)

        return pecentages

    def insert_f(files, opt, file_count):
        """
        inserts the file into the correct dicitonary based on 
        a threshold of sucessful entries
        """

        def check_fits(df, max_d, options):
            """
            Checks that the dataframe fits the available
            space in the bins - if it doesn't then moves on
            """
            works = {}
            sucess = 0

            for frame, files in options.items():
                for file in files:
                    if len(df[frame]) < max_d[frame]:
                        add_choosen(works, frame, file)

            return works

        choices = {}
        sizes = {}

        for entry in ['test','train', 'val']:
            df = files[entry]
            max_d = max_dtype[entry]
            options = check_fits(df, max_d, opt)

            choices[entry] = options
            sizes[entry] = opt_size(options)

        if sizes['val'] > 0:
            add_dict(files['val'], choices['val'])
            file_count['val'].append(sizes['val'])
        elif sizes['test'] > 0:
            add_dict(files['test'], choices['test'])
            file_count['test'].append(sizes['test'])
        elif sizes['train'] > 0:
            add_dict(files['train'], choices['train'])
            file_count['train'].append(sizes['train'])

    return fill(lang, maximum, probability_f, insert_f)


def fill(lang, maximum, prob_f, insert_f):
    """
    Fills the langauge to match the given shape
    """

    i = 0

    df = load_df([lang])[lang]

    files = {
        dtype: {key: [] for key, _ in maximum.items()}
        for dtype in ["train", "test", "val"]
    }

    file_count = {dtype : [] for dtype in ['train', 'test', 'val']}

    for index, row in list(df.iterrows()):
        opt = make_dict(row)

        if opt_size(opt) > config.NUM_SPEAKERS:
            # print("enter")
            opt = select_audio(opt, maximum, current_size(files), prob_f)

        # Logic for choosing which dict to add
        insert_f(files, opt, file_count)

        i += 1

    return files, file_count


def add_dict(main, new):
    """
    Adds new values to our larger dictionary of audio
    files
    """
    for key, value in new.items():
        main[key] += value


def current_size(files):

    d1 = files["train"]
    d2 = files["test"]
    d3 = files["val"]

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

def add_choosen(dictionary, key, val):
    """
    Adds values to the dictionary based on it's key and
    value
    """
    if key in dictionary:
        dictionary[key].append(val)

    else:
        dictionary[key] = [val]

def select_audio(options, maximum, current, prob_f):
    """
    Weighs audio file to have more longer clips
    but fills the dataset with the maximum amount of
    audio files
    """

    def remove_opt(dictionary, key, file):
        """
        Removes the link, if no longer a link in
        datastructure removes that key as well
        """
        dictionary[key].remove(file)

        if len(dictionary[key]) == 0:
            del dictionary[key]

    weights = prob_f(options, current)
    chosen_files = {}

    i = 0
    j = 0

    while (
        j < config.NUM_SPEAKERS
        and len(options) > 0
        and all([val != 0 for val in weights])
    ):
        frame, files = random.choices(list(options.items()), k=1, weights=weights)[0]
        file = random.choice(files)

        # Removes from the dictionary
        remove_opt(options, frame, file)

        # Updates weights
        weights = prob_f(options, current)

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

def generate_max_dtype(files):
    """
    Lists out the maximum by datastet type
    """
    x = {}

    for dtype, data in files.items():
        x[dtype] = {}
        for key, value in data.items():
            x[dtype][key] = len(value)

    return x
            

#############  -----------------------  ################


def create_dataset(languages):
    """
    Runs the majority of the code to create a
    """
    # Audio files will all be saved key : val
    dataset = {}
    speakers = {}

    # Finds maximum, then smallest language
    maximum = find_sizes(languages)
    smallest_language = smallest_lang(languages)
    print(f'Smallest dataset is: {smallest_language}')

    # Fills the smallest language, saves the files and 
    # largest shape each audio file can be
    audio_files, speak = fill_smallest(smallest_language, maximum)

    dataset[smallest_language] = audio_files
    speakers[smallest_language] = speak

    maximum_size_time = generate_max_dtype(audio_files)
    print(f'Finished: {smallest_language}')

    # Fills the rest of the languages
    for lang in languages:
        if lang is smallest_language:
            continue

        audio_files, speak = fill_other(lang, maximum, maximum_size_time)

        dataset[lang] = audio_files
        speakers[lang] = speak

        print(f'Finished: {lang}')

    return dataset, speakers, maximum

if __name__ == "__main__":
    lang = ["ta", "en", "es", "ja", "it", "de", "nl"]
    
    x, y = create_dataset(lang)

    with open('notebooks/play/files.pkl', 'wb') as f:
        pickle.dump(x, f)

    with open('notebooks/play/speak.pkl', 'wb') as f:
        pickle.dump(y, f)