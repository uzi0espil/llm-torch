import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch


def plot_losses(train_losses, val_losses, n_epochs=None, **kwargs):
    n_epochs = len(train_losses) if n_epochs is None else n_epochs
    epochs_tensor = torch.linspace(0, n_epochs, len(train_losses))
    fig, ax1 = plt.subplots(**kwargs)

    # Plot training and validation loss against epochs
    ax1.plot(epochs_tensor, train_losses, label="Training loss")
    ax1.plot(epochs_tensor, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    fig.tight_layout()
    plt.show()
