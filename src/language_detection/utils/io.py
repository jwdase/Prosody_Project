from pathlib import Path
import shutil
import torch


def save_test(test_loader, base):
    """
    Saves the test data for data validation later
    """
    for i, (inputs, outputs) in enumerate(test_loader):
        torch.save(inputs, f"{base}/inputs_{i}.pt")
        torch.save(outputs, f"{base}/outputs_{i}.pt")

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

def grab_device():
    """
    Ensures that we're using the GPU for training
    """
    assert torch.cuda.is_available(), "No GPU on Device"


def save_model(model, loc):
    torch.save(model.state_dict(), f'{loc}/final_model.pth')


def save_spect(spectrograms, path, batch_size, i):
    """
    Code to save the files,
    """

    def file_name(length, num):
        return (
            "batch_" + "0" * (length - len(str(num))) + str(num) + "_" + str(batch_size)
        )

    torch.save(spectrograms, Path(path + "/" + file_name(5, i) + ".pt"))