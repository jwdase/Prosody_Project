"""
This script loads data so we can run the matlab file on
it to create the prosody based audio recordings. 
"""

import shutil
import os
import json
from glob import glob
from pathlib import Path

def check_path(base):
    """
    Clears the folder that we're going to
    load the model data into
    """
    folder = Path(base)

    # Checks if the folder exists, and then deletes folder and recreates it
    if folder.is_dir():
        shutil.rmtree(base)

    folder.mkdir(exist_ok=True, parents=True)

def update_config(language, save_directory):
    """Update config.json with the save directory for a specific language"""
    # Load existing config or start fresh
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        config = {}
    
    config[language] = save_directory
    
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Updated config.json: {language} -> {save_directory}")


def get_files(start):
    """ Returns a list of all audio files in folder """

    files = glob(f"{start}/clips/*.mp3")
    print(f"{len(files)} files generated")

    return files

def write_txt(files, language):
    with open(f'data/{language}.txt', 'w') as f:
        for file in files:
            f.write(file + "\n")

def copy_validated(start, destination):
    shutil.copy(f"{start}/validated.tsv", f"{destination}/validated.tsv")


def main(lang, start, destination):
    """ Runs main loop """
    files = get_files(start)                                # Generate a list of files
    write_txt(files, lang)                              # Write them out
    update_config(lang, f"{destination}/clips")         # Add root to config
    check_path(f"{destination}/clips")                      # Ensures path exists
    copy_validated(start, destination)                      # Loads validated .tsv for simplify

if __name__ == "__main__":
    language = ["ja", "uz", "it"]

    for lang in language:
        start = f"/orcd/archive/evelina9/001/u/moshepol/om2/prosody/data/raw_audio/{lang}"

        # Ensure that target is reachable
        destination = f"/orcd/archive/evelina9/001/u/jwdase/Augmented_Data/{lang}"

        main(lang, start, destination)

        print(f"Complete: {lang}")