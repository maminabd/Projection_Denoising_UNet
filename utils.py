
import matplotlib.pyplot as plt
import torch


def get_device():
    
    """
    Select CUDA device if available and print GPU info, otherwise use CPU.

    Args:
        None

    Returns:
        torch.device: "cuda" if a GPU is available, else "cpu".
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available, using CPU")
    return device


def plot_history(train_losses, val_losses, learning_rates, filename):
    
    """
    Save a picture of 2 plots: 1- training/validation loss wrt epochs and 2- learning rate wrt epochs

    Args:
        train_losses: list of the training loss of each epoch.
        val_losses: list of the validation loss of each epoch.
        learning_rates: list of the learning rate of each epoch.
        filename: name of picture of the 2 plots to be saved.
    """
    
    epochs_range = range(1, len(train_losses)+1)
    last_epoch = len(train_losses)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # training/validation loss
    ax1.plot(epochs_range, train_losses, label='Train Loss', linewidth=2)
    ax1.plot(epochs_range, val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss (up to epoch {last_epoch})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Learning rate (log scale)
    ax2.plot(epochs_range, learning_rates, label='Learning Rate', linewidth=2, color='purple')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title(f'Learning Rate Schedule (up to epoch {last_epoch})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close(fig)


def save_checkpoint(path, model, optimizer, scheduler, epoch, **extra):
    """
    Save model, optimizer and scheduler states plus any extra information.

    Args:
        path: output .pth file.
        model, optimizer, scheduler: Objects whose state_dict is saved.
        epoch: index (0-based) of the last completed epoch.
        **extra: additional entries stored in the checkpoint (losses, config...).
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    checkpoint.update(extra)
    torch.save(checkpoint, path)
