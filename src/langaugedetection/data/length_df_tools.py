NUM_SPEAKERS_RECORD = 4

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


def select_array(df, key):
    '''
    Returns all links that work for a certain 
    language with-in that timeframe
    '''
    
    test_array = []
    validation_array = []
    train_array = []

    for i, x in enumerate(df[key]):

        try:
            val = convert_to_list(x)[0:NUM_SPEAKERS_RECORD]
        except IndexError:
            val = convert_to_list(x)

        if i % 10 == 0:
            test_array.extend(val)
        elif i % 10 == 1:
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


