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
    array = []
    for x in df[key]:
        array.extend(convert_to_list(x))

    return array

def key_length(df, key):
    '''
    Takes in a df and a certain key
    and adds up all the values of that column
    '''

    return len(select_array(df, key))


