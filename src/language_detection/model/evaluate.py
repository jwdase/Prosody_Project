import matplotlib.pyplot as plt

def plot_loss(total_loss, validation_loss, base):
    """
    Plots Loss over time
    """
    x = range(len(total_loss))

    plt.plot(x, total_loss, label="Train Loss", color="blue")
    plt.plot(x, validation_loss, label="Validation Loss", color="orange")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Epochs vs. Loss")  # fixed typo
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{base}/loss_graph.png")