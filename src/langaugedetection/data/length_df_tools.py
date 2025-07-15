import random

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

    return len(select_array(df, key))


