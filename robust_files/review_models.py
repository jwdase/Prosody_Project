'''
Script that records accuracy, classification report, and 
confusion matrix, puts in folder labeled 'review' in sub 
directory - saved by model training statistics
'''

import glob
import os
import torch
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


from path import setup_project_root
setup_project_root()

from train_gpu import LanguageDetector


def gather_models(base):
    '''
    Discovers all the available models to test
    '''

    return [p for p in glob.glob(f'{base}*') if os.path.isdir(p)]

def build_tensor(base):
    '''
    Loads the input tensors then turns them
    into a training loader
    '''

    inputs = glob.glob(f'{base}/inputs*.pt')
    labels = glob.glob(f'{base}/outputs*.pt')

    print(f'Total Inputs: {len(inputs)}')
    print(f'Total Outputs: {len(labels)}')

    tensors = []

    first = {'train' : {'en' : ''}}

    for i, (inp, out) in enumerate(zip(inputs, labels)):

        if i % 5 == 0:
            print(inp.split('/')[-1])
            print(out.split('/')[-1])

        if i == 0:
            first['train']['en'] = torch.load(inp, weights_only=False)

        tensors.append(
            TensorDataset(
                torch.load(inp, weights_only=False), 
                torch.load(out, weights_only=False)
                )
            )
    
    loader = DataLoader(ConcatDataset(tensors), batch_size=256, num_workers=4, shuffle=False)

    return loader, first

def load_model(base, arch, enc):
    '''
    Loads to model and returns it 
    '''
    num_classes = len(enc.classes_)

    model = LanguageDetector(num_classes, arch['train']['en'][0].shape)
    model.load_state_dict(torch.load(f'{base}/best_model.pth', weights_only=True))
    
    model.eval()

    model.to('cuda')

    return model

def test(model, loader):
    '''
    Runs through all the values testing the model
    '''
    y_pred = []
    y_true = []

    with torch.no_grad():
        for inputs, label in loader:

            inputs = inputs.to('cuda')
            label = label.to('cuda')

            inputs = inputs.unsqueeze(1)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, dim=1)

            y_pred.append(predicted)
            y_true.append(label)

    y_pred = torch.cat(y_pred).cpu()
    y_true = torch.cat(y_true).cpu()

    return y_pred, y_true

def write_acc(direct, name, y_true, y_pred):
    '''
    Creates a .txt file with accuracy
    '''
    acc = f"Accuracy: {accuracy_score(y_true, y_pred)}"

    with open(f'{direct}/{name}.txt', 'w') as f:
        f.write(acc)

    return None

def write_class_report(direct, name, y_true, y_pred, enc):
    '''
    Creates the classification Report
    '''
    report = classification_report(y_true, y_pred, target_names=enc.classes_, output_dict=True)

    df = pd.DataFrame(report)
    df.to_csv(f'{direct}/{name}.csv')

    return None

def write_confusion_matrix(direct, name, y_true, y_pred, enc):
    '''
    Creates the confusion matrix
    '''
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=enc.classes_)
    disp.plot(xticks_rotation='horizontal')
    plt.savefig(f'{direct}/{name}.png')

    return None


def main(base, store):
    '''
    Runs through the code
    '''

    for opt in gather_models(base):

        # Ensures Directory exists
        try:
            loader, architecture = build_tensor(opt)
        except AssertionError:
            print(f'No directory: {opt}')
            continue

        encoder = joblib.load(f'{opt}label_encoder.pkl')
        model = load_model(opt, architecture, encoder)
        y_pred, y_true = test(model, loader)

        title = opt.split('/')[-1]
        inputs = (store, title, y_true, y_pred)

        write_acc(*inputs)
        write_class_report(*(inputs + (encoder,)))
        write_confusion_matrix(*(inputs + (encoder,)))


if __name__ == '__main__':
    
    parent_dir = '/om2/user/moshepol/prosody/models/'
    store_dir= 'review'

    main(parent_dir, store_dir)