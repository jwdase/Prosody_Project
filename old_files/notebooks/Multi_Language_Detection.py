#!/usr/bin/env python
# coding: utf-8
import random


# Set main root 
from setup import setup_project_root
setup_project_root()


from langaugedetection.data.com_voice_dir import open_files, person_to_group
from langaugedetection.data.audio_length import valid_paths, random_audio
 
def load_languages(languages, choices = 10_000):
    '''
    For each language opens a file called /{lang}/validated.tsv
    that contains speaker id and all the validated audio files
    then for each language it transforms the data to dictionary mapping
    each speaker id to a list of valid audio files it has
    '''

    files = {lang : open_files(lang, choices) for lang in languages}
    # Loads a csv for each language that contains x 
    # datapoints of audio recording, if unspecified
    # then no datapoints

    lang_people = {lang : person_to_group(df) for lang, df in files.items()}
    # Inside each dataframe makes a dictionary mapping
    # Each unique speaker to all the audio files
    # They have produces

    for lang, groups in lang_people.items():
        print(f'Unique {lang} speakers: {len(groups)}')

    audio_per_person = 2

    lang_people_to_paths = {}

    for lang, lang_dict in lang_people.items():

        lang_people_to_paths[lang] = (
            {people : random.sample(x := valid_paths(df, 5, 1, lang), min(len(x), audio_per_person)) for people, df in lang_dict.items()}
        )

        print(f'Language: {lang} complete')

    return lang_people_to_paths



def train_test_spit(lang_people, fraction = .8):
    '''
    Chooses a split in people between training and testing data,
    we do this by choosing the speaker and thne selecting the speaker's
    audio
    '''

    def chooses_speaker(d, fraction = .8):
        '''
        Splits the dataset to be 80% of training
        '''

        def flatten(array):
            '''
            The datastructure that each key points to [[val, val], val] 
            and we need to flatten to [val, val, val]. That is 
            '''
            x = []
            for x_val in array:
                if isinstance(x, list):
                    x.extend(x_val)
                else:
                    x.append(x_val)
            return x

        train_keys = set(random.sample(list(d.keys()), int(.8 * len(d.keys()))))
        test_keys = set(d.keys()) - train_keys

        train = flatten(train_keys)
        test = flatten(test_keys)

        return train, test


    lang_train = {}
    lang_test = {}

    for lang, dictionary in lang_people_to_paths.items():
        train, test = chooses_speaker(dictionary)

        lang_train[lang] = train
        lang_test[lang] = test

    for (lang, train_item), (lang, test_item) in zip(lang_train.items(), lang_test.items()):
        print(f'Language: {lang}, Training: {len(train_item)}, Testing: {len(test_item)}')

    return lang_train, lang_test



from langaugedetection.data.spectrogram import parse
from langaugedetection.data.tensor_construct import build_tensor
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, ConcatDataset
import torch

def shape_tensor(lang_train, lang_test):
    '''
    Builds the train and test tensors and then returns this
    in the process 
    '''

    train_choices = min([len(item) for _, item in lang_train.items()])
    test_choices = min([len(item) for _, item in lang_test.items()])



    lang_short_train = {}
    lang_short_test = {}

    for (lang, train_item), (lang, test_item) in zip(lang_train.items(), lang_test.items()):
        lang_short_train[lang] = random.sample(train_item, train_choices)
        lang_short_test[lang] = random.sample(test_item, test_choices)


    train_tensor = {}
    test_tensor = {}

    for (lang, train_item), (lang, test_item) in zip(lang_short_train.items(), lang_short_test.items()):
        train_tensor[lang] = build_tensor(parse(train_item))
        test_tensor[lang] = build_tensor(parse(test_item))


    for lang, train in train_tensor.items():
        print(f'{lang} tensor shape is: {train.shape}')

    encoder = LabelEncoder()
    encoder.fit(list(train_tensor.keys()))

    train_label_array = {lang : encoder.transform([lang] * train.shape[0]) for lang, train in train_tensor.items()}
    test_label_array = {lang : encoder.transform([lang] * test.shape[0]) for lang, test in test_tensor.items()}

    print(f'Classes: {encoder.classes_}')


    train_datasets = []
    for (lang, train), (lang, label) in zip(train_tensor.items(), train_label_array.items()):
        train_datasets.append(TensorDataset(train, torch.tensor(label, dtype = torch.long)))


    test_datasets = []
    for (lang, test), (lang, label) in zip(test_tensor.items(), test_label_array.items()):
        test_datasets.append(TensorDataset(test, torch.tensor(label, dtype = torch.long)))


    train = ConcatDataset(train_datasets)
    test = ConcatDataset(test_datasets)

    print(f'Training values: {len(train)}')
    print(f'Testing values: {len(test)}')

    return train, test



import torch.nn as nn
def build_model(languages):
    '''
    Returns an initialized nn
    '''
    class LanguageDetector(nn.Module):
        def __init__(self):
            super().__init__()

            # 2 Convolution Layers
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

            # Neuron Layers
            # Now apply linear layer --> Dimensions of input are 
            # Con1 
            # (1025, 216) --> (512, 108)
            # Con2
            # (512, 108) --> (256, 54)
            # Now we have 32 of these with respective filters applied
            self.fc1 = nn.Linear(32 * 256 * 54, 256)
            self.fc2 = nn.Linear(256, 32)
            self.fc3 = nn.Linear(32, len(languages))

            # Relu, Pool Function
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)


        def forward(self, x):

            # 2D Convolution Apllication
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))

            # Flatten Dimensions
            x = x.view(x.size(0), -1)

            # Dense Layers
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)

            return x

    model = LanguageDetector()

    return model


from torch.utils.data import DataLoader

def run_model(model, train):
    '''
    Creates the dataloader then initializes training phase
    '''

    loader = DataLoader(train, batch_size=64, shuffle = True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = .001)



    # Train
    model.train()
    num_epochs = 1 # Should be around 25
    total_loss = []

    for i in range(num_epochs):

        epoch_loss = 0

        for x_batch, y_batch in loader:

            # Evaluate
            outputs = model(x_batch).squeeze(1)
            loss = criterion(outputs, y_batch)

            # Update Model
            model.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss

        total_loss.append(epoch_loss / len(loader))

        print(f'Epoch : [{i+1} /{num_epochs}], loss {total_loss[-1]}')

    return model, total_loss

def save(model, test, total_loss):
    '''
    Takes in the 3 datafiles and saves them in their respective folers
    '''

    # Save output, Training Loss
    with open('models/model1.txt', 'w') as f:
        for i, loss in enumerate(total_loss):
            f.write(f'Epoch {i}: {loss:.4f} loss\n')

    # Save output, Testing Tensor
    torch.save(test, 'models/test.pt')

    # Save model
    torch.save(model.state_dict(), 'models/model1.pth')

def run_and_save(languages):
    '''
    Code to run the whole program
    '''

    lang_people_to_paths = load_languages(languages)
    lang_train, lang_test = train_test_spit(lang_people_to_paths)
    train, test = shape_tensor(lang_train, lang_test)
    model = build_model(languages)
    model, total_loss = run_model(model, train)
    save(model, test, total_loss)

if __name__ == '__main__':
    languages = ['en', 'it', 'es', 'de']
    run_and_save(languages)





