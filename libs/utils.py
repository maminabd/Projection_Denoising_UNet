
import logging
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: plots are only saved to file
import matplotlib.pyplot as plt
import torch

logger = logging.getLogger(__name__)

CHANNEL_NAMES = ('LE', 'HE')  # names of the 2 channels (Low Energy and High Energy) of the projections


def get_device(): # pragma: no cover
    
    """
    Select CUDA device if available and log GPU info, otherwise use CPU.

    Args:
        None

    Returns:
        torch.device: "cuda" if a GPU is available, else "cpu".
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("CUDA Version: %s", torch.version.cuda)
        logger.info("GPU Memory: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
    else:
        logger.info("CUDA not available, using CPU")
    return device


def plot_history(train_losses, val_losses, learning_rates, start_epoch, filename):  # pragma: no cover
    
    """
    Save a picture of 2 plots: 1- training/validation loss wrt epochs and 2- learning rate wrt epochs

    Args:
        train_losses: list of the training loss of each epoch.
        val_losses: list of the validation loss of each epoch.
        learning_rates: list of the learning rate of each epoch.
        start_epoch: number of the epoch of the first element of the lists (the x axis starts from it).
        filename: name of picture of the 2 plots to be saved.
    """
    
    epochs_range = range(start_epoch, start_epoch + len(train_losses))
    last_epoch = start_epoch + len(train_losses) - 1

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


def absolute_difference(denoised, ground_truth):
    """
    Pixel-wise absolute difference between a denoised projection and its ground truth.

    Args:
        denoised: Tensor of shape (C, H, W).
        ground_truth: Tensor of shape (C, H, W).

    Returns:
        Tensor of shape (C, H, W) with the absolute differences.
    """

    return (denoised - ground_truth).abs()


def mean_absolute_difference(denoised, ground_truth):
    """
    Mean absolute difference between a denoised projection and its ground truth, for each channel.

    Args:
        denoised: Tensor of shape (C, H, W).
        ground_truth: Tensor of shape (C, H, W).

    Returns:
        List with the mean absolute difference of each channel, i.e. [mean_LE, mean_HE].
    """

    difference = absolute_difference(denoised, ground_truth)

    # averages over the last two dimensions i.e. H and W of tensor of shape (C, H, W)
    # and converts the resulting tensor to a Python list
    return difference.mean(dim=(1, 2)).tolist()


def projs_comparison(noisy_tensor, denoised, ground_truth, difference, mean_differences, filename, title): # pragma: no cover
    """
    Save a picture comparing the denoised projection with the ground truth.

    The picture has 2 rows and 4 columns. Each row corresponds to a channel (LE, HE).
    The first column shows the noisy projection,
    The second column shows the denoised projection,
    The third column shows the ground truth (noiseless projection),
    The fourth column shows the absolute difference map between the denoised projection and the ground truth.

    Args:
        noisy_tensor: Tensor of shape (C, H, W).
        denoised: Tensor of shape (C, H, W).
        ground_truth: Tensor of shape (C, H, W).
        difference: Tensor of shape (C, H, W), absolute difference between denoised and ground truth.
        mean_differences: List with the mean of the absolute difference of each channel.
        filename: Output picture file.
        title: Title of the whole picture.
    """

    fig, axes = plt.subplots(2, 4, figsize=(18, 11), squeeze=False)

    for row, channel_name in enumerate(CHANNEL_NAMES):
        panels = [
            (noisy_tensor, f"Noisy ({channel_name})"),
            (denoised, f"Denoised ({channel_name})"),
            (ground_truth, f"Ground truth ({channel_name})"),
            (difference, f"Absolute difference ({channel_name}), mean = {mean_differences[row]:.6f}"),
        ]
        for column, (tensor, panel_title) in enumerate(panels):
            image = axes[row, column].imshow(tensor[row].numpy(), cmap='gray')
            axes[row, column].set_title(panel_title)
            fig.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=100, bbox_inches='tight')
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
        'checkpointed_epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    checkpoint.update(extra)
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer, scheduler, device):
    
    """
    Restore the model, optimizer and scheduler states from a checkpoint.

    Args:
        path: Checkpoint (.pth) file path written by save_checkpoint function.
        model, optimizer, scheduler: Objects whose state is restored in place.
        device: Device the checkpoint tensors are loaded onto (CPU or GPU).

    Returns:
        Tuple of (checkpointed_epoch, checkpointed_val_loss): respectively are:
            -the epoch to resume from after loading the checkpoint.
            -the validation loss stored in the checkpoint (None if the checkpoint has no 'val_loss' entry),
             which acts as a reference for the best validation loss so far.
    """

    # loads a checkpoint (weights_only=True: the checkpoint only holds tensors and numbers)
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    # take the parameters of the saved model from the dictionary 
    model_param = checkpoint['model_state_dict']
    optimizer_param = checkpoint['optimizer_state_dict']
    scheduler_param = checkpoint['scheduler_state_dict']
    checkpointed_epoch = checkpoint['checkpointed_epoch'] + 1 # resume from the next epoch after the checkpointed one
    checkpointed_val_loss = checkpoint.get('val_loss') # .get() is used because 'val_loss' is an optional entry, stored by **extra of save_checkpoint.

    # load those parameters into their corresponding objects.
    model.load_state_dict(model_param)
    optimizer.load_state_dict(optimizer_param)
    scheduler.load_state_dict(scheduler_param)

    return checkpointed_epoch, checkpointed_val_loss
