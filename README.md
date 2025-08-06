# Prosody Research

## Importance

**Simple Goal:** To classify languages based on prosodic features. \
**Importance:** We understand how the languages of the world differ in syntax (SVO vs. SOV orders); phonology (which languages have similar phonemmes); however, we do not have a cmomplete understanding of how prosody differs language to language. This could be because prosody is not important or it could also be becuse we don't yet have the vocabulary to understand prosody. **The goal** of this research is to develop that vocablary.

## Installations and Instructions

### 1. Install Poetry If you haven't already installed [Poetry]
run: ```pip install poetry``` 

### 2. Clone the repository and install dependencies 
```bash git clone https://github.com/jwdase/Prosody_Project.git``` \
```cd langaugedetection poetry install ```

### 3. Activate the virtual environment (optional but recommended) 
```bash poetry shell ```

## Usage

### Importing a New Language
This project works on CommonVoice Languages dowloaded into this folder: */om2/user/moshepol/data/raw_audio*, so we need to upload all new languages to that folder.

#### Steps
1. Download language from CommonVoice onto laptop
1. Transfer language audio *(still compressed)* from laptop to */om2/user/moshepol/data/raw_audio* using this command: \
```rsync -avhP ~/local_folder/ login:/om2/user/moshepol/data/raw_audio/```
1. Extract audio files using: \
```tar -xvzf my_audio.tar.gz```
1. Move file to correct directory: \
```mv my_audio /om2/user/moshepol/data/raw_audio/```
1. Create a speaker to audio length dataframe from this audio.  To create the dataframe for a language, write the name of language folder in the ```scripts/lengths.py``` ```__main__```, then run: \
```PYTHONPATH=. python scripts/lengths.py```

Notes:
- This process should take anywhere from 5 minutes to 1 hour depending on the number of audio files in the language.
- We make this dataframe because it is a slow process and creating the dataframe each time we want to apply some transformation on the audio would be costly (7 languages, 7 hours)
- The dataframe caches information on which audio files each speaker produces and their respective lengths. 
- The rows are speakers the columns are lengths with links to audio files. 
- The value in each location is the list of audio files for the speaker connected by hyphens. To remove the hypens and convert files into a list use ```convert_to_list``` from ```csv_tools.read```.

### Creating a Spectrogram
Creating a spectrogram is significantly faster on GPU, thus this code works on GPU. If you do not have acess to a GPU, in ```language_detection.config``` change ```DEVICE``` from ```GPU``` to ```CPU```. 

#### Parameters
1. Languages - Enter a list of languages (downloaded and already applied all the importing a New Language steps)
1. entry - entry is a dictionary that specifies how it wants a spectrogram created. The first properties
    - sr - Sample Rate (measured in Hz)
    - n_fft - Number of samples per frame in computing Short-Time Fourier Transform (STFT)
    - hop_length - Number of samples to move between STFT 
    - spect_function - How we want to compute the spectrogram. Right now we have two methods in the ```spectrogram.function``` folder. The first one ```compute_spectrogram_batch``` computes a spectrogram over all frequencies. The second one ```compute_lowpass_spectrogram_batch``` computes a spectrogram only using frequencies under a limit - specified in ```config```.
1. location - tells us where to save the spectrograms - once created

#### Running Spectrogram Example

Step 1: Create the file in scripts folder or notebooks. Fill with these values - modifying for specific use. 

```
python

from language_detection.spect import main
from language_detection.data.spectrogram.functions import compute_spectrogram_batch

languages = ["en", "it", "es", "de", "nl", "ta", "ja", "tr", "uz"] 

location = "/om2/user/moshepol/prosody/data/low_pass_data" 

n_fft = 1024 
hop_length = 512 
sr = 16_000
spect_f = compute_spectrogram_batch

entry = { "sr": sr, "n_fft": n_fft, "hop_length": hop_length, "spect_f": spect_f} 

main(languages, entry, location) 
```

Step 2: Run file
```
PYTHONPATH=. python folder/file_name.py
```

#### Files Created
- In the directory specified spectrograms will be saved in this method
    - *specified*/{lang}/{spect}/{train}/{spectrograms.pt}
    - There will be many of them and they will be numbered
- Each spectrogram file contains 32 spectrograms. The data is organized as this:  
```
spect = {
    'spect' : SPECTROGRAMS
    'duration' : HOW LONG EACH AUDIO FILE IS
}
```

- In addition, 3 other files will be created
    1. dataset - list of audio files used
    1. speakers - how many audio files each speaker gave
    1. lengths - the breakdown of different length audio files in dataset

    Note: You can create graphs with this data in the plots folder

#### Values in Config
Some values in Config influence how we create the spectrogram. Here are the important ones. 
- NUM_SPEAKER - States the maximum number of audio recordings per a speaker, usually less
- MAX_LENGTH - Maximum length of audio file (in seconds) 
- WORKERS - Num Workers preparing data for FFT
- CUTOFF - Maximum frequency for low pass filer (low pass filter removes all audio in frequencies higher than the cutoff)
- WEIGHTS - Develops the priority queue for which length audio files should be added to our dataset from the whole commonvoice dataset. Put higher numbers in audio file lengths you think are more important. (You can view language audio file length distribution in ```plots.distribution```)

### Training a Model
Once you create the spectrograms, you can then train the model. There are a variety of models to choose from, they're all saved under ```model.networks```. The expectation for a model is that in it's forward pass it takes in both the spectrogram and the length of the audio file. If it's ```forward``` function doesn not take those values there will be an error.

#### Parameters
1. languages - A list of languages we want to train the model on
1. model - The model we want to use for language detection
1. origin - Where the spectrograms were saved - loads the data
1. base - Where we want to save the testing data, model, and graphs about training.


#### Running Model Example
Step 1: Create a file in scripts or notebooks folder with the following lines. Modify the values given for specific use

```
from language_detection.train import main
from language_detection.model.networks import VarCNNTransformerLanguageDetector

language = ["en", "de", "nl", "es", "it", "ja", "ta", "tr", "uz"]

origin = '/om2/user/moshepol/prosody/data/low_pass_data'
base = '/om2/user/moshepol/prosody/models/test/prosody_epoch_20_tr/'

main(language, VarCNNTransformerLanguageDetector, origin, base)
```

Step 2: Run the file you just made
```
PYTHONPATH=. python directory/my_file.py
```

#### Saved Values
- The best model will be saved as long as the last model
- All the testing tensors will be saved as well
- There will be two plots 
    1. Plot of learning rate over epoch
    1. Plot of validation loss, training loss over epoch

#### Config Controls
Some of the parameters used in training are acessed through ```config.py```. This is because we don't want to have to pipe hundreds of different values into the functions in order for them to work correctly. In addition, the values in config shouldn't change much. Here is an overview of the important ones for training.

1. BATCH_SIZE - Keep below 256 because we cut any unfilled batch from trainng. Thus a batch of only 255 audio files would not be trained on. Sometimes if there is a **divide by zero error** it is triggered by batch_size being to large. 
1. NUM_WORKER - In the loader specifies how many cores should be devoted to prepating data to be transfered to GPU
1. NUM_EPOCHS - Amount of training rounds
1. LR - How far each learning step should be 
1. PATIENCE - How long to wait with no improvement on validation loss to lower LR
1. FACTOR - How much to reduce LR by when little improvement in validation loss
1. THRESHOLD - Threshold of loss decrease to trigger LR decrease
1. ERROR - How much better the new model has to be on validation accuracy to save it

### Reviewing the Model
Go to file ```plots.review_model``` and paste directory the you saved the model into as the ```base```. Note, remove last '/'. Then, set name to be the name you want for all your audio files. Click ```Run All``` and your plots will be created in the ```plots.model``` folder. 


## Main files
. \
├── language_detection \
│   ├── config.py # Defauls located\
│   ├── data # Folder for data procesing\
│   ├── __init__.py\
│   ├── model # Folder for loading model data, and model\
│   ├── __pycache__\
│   ├── spect.py # Spectrogram making function\
│   ├── train.py # Training function\
│   └── utils # Folder for useful helpers\
├── notebooks\
│   └── play # Put any experiment files here\
├── plots\
│   ├── models # Saves all model review files\
│   ├── figures # Saves all figures made\
│   ├── distributions.ipynb # How many files per length per language\
│   ├── freq_time.ipynb # Checked different time lengths for a window\
│   ├── review_model.ipynb # Used to review model\
│   ├── speakers.ipynb # Tells how many unique speakers in each dataset\
│   ├── training_distribution.ipynb # Reviews distribution of lengths which model trained on \  
│   └── unique_speakers.ipynb # gets number of unique speakers\
├── scripts\
│   ├── language.py # Language Detection\
│   ├── lengths.py # Generates the lengths database\
│   └── prosody.py # Low Pass Filter Language Detection\

11 directories, 15 files