import matplotlib.pyplot as plt
import numpy as np

def plot_loss(total_loss, validation_loss, base):
    """
    Plots Loss over time
    """

    x = range(len(total_loss))

    plt.figure()
    plt.plot(x, total_loss, label="Train Loss", color="blue")
    plt.plot(x, validation_loss, label="Validation Loss", color="orange")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Epochs vs. Loss")  # fixed typo
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{base}/loss_graph.png")
    plt.close()

def plot_lr(lr, base):
    """
    Plots the learning rate over time
    """
    x = range(len(lr))

    plt.figure()
    plt.plot(x, lr)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epoch vs. Loss")
    plt.grid(True)
    plt.savefig(f'{base}/lr_graph.png')
    plt.close()