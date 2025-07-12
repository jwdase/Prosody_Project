"""
This Script makes a language detection model, it is trained
on spectrograms which must be first created in the make_spect file
"""


from path import setup_project_root
setup_project_root()

import torch
import glob
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
import torch.nn as nn
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

def grab_device():
    '''
    Ensures that we're using the GPU for training
    '''
    assert torch.cuda.is_available(), 'No GPU on Device'

def generate_encoder(languages):
    '''
    Creates the encoder for the respective languages
    '''

    encoder = LabelEncoder()
    encoder.fit(list(languages))

    print(f'Classes: {encoder.classes_}')

    return encoder

def load_audio(languages, window):
    '''
    Loads the audio files from the directory specified below
    stores them as a dict
    key: ['train']
    value: dict (key: lang, item: tensors)
    '''
    data_types = ['train', 'test', 'validation']

    # Create the array which saves data-values
    # References allows us to manually verify the audio is 
    # from the correct language
    tensors = {dtype : {} for dtype in data_types}
    reference = {dtype : {} for dtype in data_types}

    for lang in languages:
        for dtype in data_types:
            base = f'/om2/user/moshepol/prosody/data/raw_audio/{lang}/spect/{dtype}/{window}/'

            paths = glob.glob(base + '*.pt')

            first_audio = True

            spectograms_tensors = []
            
            for path in paths:
                x = torch.load(path, weights_only=False)

                if first_audio:
                    reference[dtype][lang] = x[0]
                    first_audio = False

                # Put Tensor on range 0 --> 1
                spectograms_tensors.append(x)

            # Saves each languages tensor
            tensors[dtype][lang] = spectograms_tensors.copy()

    return tensors, reference


def standardize_tensors(tensors):
    '''
    Normalize each audio file giving them 
    their respective z-score
    '''

    # Calculates tensor using 'Training data'
    train_tensors = [tensor for lang_tensors in tensors['train'].values() for tensor in lang_tensors]
    mean, std = calc_mean(train_tensors)

    # Normalize 'Training Data'
    for use, dataset in tensors.items():
        for lang, data in dataset.items():
            tensors[use][lang] = [(d - mean) / std for d in data]

    return tensors

def calc_mean(tensor_list):
    '''
    Returns the mean and standard deviation
    of each tensors
    '''
    sum_ = 0.0
    sum_sq = 0.0
    count = 0

    for tensor in tensor_list:  # each tensor is e.g. [64, 1025, 172]
        sum_ += tensor.sum()
        sum_sq += (tensor ** 2).sum()
        count += tensor.numel()

    mean = sum_ / count
    std = (sum_sq / count - mean ** 2).sqrt()

    return float(mean), float(std)

def print_data_mean(tensors):
    '''
    Prints mean for tensors to ensure data
    is normalized
    '''

    for use, dataset in tensors.items():
        mean = 0.0
        std = 0.0

        for lang, data in dataset.items():
            mean_, std_ = calc_mean(data)

            mean += mean_
            std += std_

        print(f'{use} has mean: {mean / len(dataset)}, std: {std / len(dataset)}') 


def flatten_data(tensors, encoder):
    '''
    Takes in the tensors dataset - now normalized with
    dict hiercachy and turns it into datasets
    '''
    result = {
        'train' : None,
        'test' : None,
        'validation' : None
    }

    for dtype, dataset in tensors.items():

        data_sets = []
        
        for lang, data in dataset.items():

            # Calculates the number of labels to assign
            values = len(data) * data[0].shape[0]

            x = [d for d in data]
            y = encoder.transform([lang] * values)

            data_sets.append(
                TensorDataset(torch.cat(x, dim=0), torch.tensor(y, dtype=torch.long))
                )

        result[dtype] = data_sets

    return result


def build_loaders(tensors, batch_size=64, num_workers=1):
    '''
    Creates out dataloaders for training
    '''

    def load_data(data, shuffle):
        return DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle = shuffle)


    train = ConcatDataset(tensors['train'])
    test = ConcatDataset(tensors['test'])
    validation = ConcatDataset(tensors['validation'])

    train_loader = load_data(train, True)
    test_loader = load_data(test, False)
    validation_loader = load_data(validation, False)

    return train_loader, test_loader, validation_loader


class LanguageDetector(nn.Module):
    def __init__(self, num_classes, middle_neuron):
        super().__init__()

        # Convolution Layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Con1: (513, 344) --> (256, 172)
        # Con2: (256, 172) --> (128, 86)

        # Neuron Layers
        self.fc1 = nn.Linear(int(middle_neuron * 32 * (1/2)**4), 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Normalize Layers
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)

        # Pooling Features
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=.3)

    def forward(self, x):

        # Apply 2D Convolution
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))

        # Flatten Dimensions
        x = x.view(x.size(0), -1)

        # Dense Layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

def calc_middle_neuron(references):
    '''
    Calculates how many neurons are 
    in the transition neuron
    '''
    product = 1
    for val in references['train']['en'].shape:
        if val % 2 == 0:
            product *= val
        else:
            product *= (val - 1)

    return product


def train_loop(model, train_loader, val_loader, num_epochs, base):
    '''
    Sets up complete training loop
    '''

    #----------------------------- Defines All parameters
    device = 'cuda'

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=.5, threshold = 1e-4
    )

    #----------------------------- Training Begins Below

    total_loss = []
    validation_loss = []
    validation_accuracy = []
    best_acc = 0.0


    for i in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_train = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Shape [64, 1025, 172] --> [64, 1, 1025, 172]
            inputs = inputs.unsqueeze(1)

            # Forward Pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backprop + Optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Tracking loss
            running_loss += loss.item() * inputs.size(0)

            # Tracks total number of values in train
            total_train += labels.size(0)

        total_loss.append(running_loss / total_train)

        model.eval()
        running_val_loss = 0.0
        correct = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                inputs = inputs.unsqueeze(1)

                # Calculates models predictions
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Solves loss
                running_val_loss += loss.item() * inputs.size(0)

                # Checks correct entries
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

                # Tracks total number of values in validation
                total_val += labels.size(0)
            
            
        validation_loss.append(running_val_loss / total_val)
        validation_accuracy.append(correct / total_val)

        if validation_accuracy[-1] > best_acc:
            best_acc = validation_accuracy[-1]
            torch.save(model.state_dict(), f'{base}/best_model.pth')
                
        scheduler.step(running_val_loss)

        if i % 5 == 0:
            print(f"Epoch [{i+1}/{num_epochs}]")
            print(f"  Train loss:      {total_loss[-1]:.4f}")
            print(f"  Validation loss: {validation_loss[-1]:.4f}")
            print(f"  Validation acc:  {validation_accuracy[-1]:.4f}")

    return total_loss, validation_loss


def plot_loss(total_loss, validation_loss, base):
    '''
    Plots Loss over time
    '''
    x = range(len(total_loss))

    plt.plot(x, total_loss, label='Train Loss', color='blue')
    plt.plot(x, validation_loss, label='Validation Loss', color='orange')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Epochs vs. Loss')  # fixed typo
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{base}loss_graph.png')


def save_test(test_loader, base):
    '''
    Saves the test data for data validation later
    '''
    for i, (inputs, outputs) in enumerate(test_loader):
        torch.save(inputs, f'{base}inputs_{i}.pt')
        torch.save(outputs, f'{base}outputs_{i}.pt')


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


def main(languages, window, num_epochs, base):
    '''
    Main code run through
    '''

    grab_device()
    print('Using Cuda')

    check_path(base)
    print('Folder Generated')

    encoder = generate_encoder(languages)
    print('Encoder Made')

    tensors, reference = load_audio(languages, window)
    print('Data Loaded')

    tensors = standardize_tensors(tensors)
    print_data_mean(tensors)
    print('Normalized Tensors')

    new_tensors = flatten_data(tensors, encoder)
    train, test, val = build_loaders(new_tensors, batch_size=256, num_workers=8)
    print('Data Loaders Constructed')

    middle_neuron = calc_middle_neuron(reference)
    model = LanguageDetector(len(languages), middle_neuron)
    print('Model Formed')

    print('Training Begun')
    total_loss, val_loss = train_loop(model, train, val, num_epochs, base)
    print('Training Ended')

    plot_loss(total_loss, val_loss, base)

    torch.save(model.state_dict(), f'{base}/final_model.pth')
    print('Model Saved')

    save_test(test, base)
    print('Saved Test Values')



if __name__ == '__main__':
    languages = ['en', 'es', 'it', 'de']
    window = 'range_5_5-6_0'
    num_epochs = 35


    base = '/om2/user/moshepol/prosody/models/n_fft=1024_hop_length=256/'

    main(languages, window, num_epochs, base)
