import pandas as pd

from language_detection.data.csv_tools.directory import (
    open_files, person_to_group,
    make_into_list, form_dataframe
)

from language_detection.data.audio import df_values
from language_detection.utils.io import check_path

def main(lang):
    """
    Runs through main code
    """

    files = person_to_group(open_files(lang))

    name, person_data = files.popitem()

    df = form_dataframe(
        name,
        df_values(person_data, lang)
    )

    for i, (name, urls) in enumerate(files.items()):
        df.loc[i] = make_into_list(
            name,
            df_values(urls, lang)
        )

    location = f'/om2/user/moshepol/prosody/data/raw_audio/{lang}/custom'
    check_path(location)
    df.to_csv(location + '/length.csv')

    print('Done')

if __name__ == '__main__':
    main('ja')

