
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
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\nCUDA not available, using CPU")
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


def projs_comaprison(denoised, ground_truth, filename, title):
    """
    Save a picture comparing the denoised projection with the ground truth.

    The picture has 2 rows and 3 columns. Each row corresponds to a channel (LE, HE).
    The first column shows the denoised projection,
    The second column shows the ground truth (noiseless projection),
    The third column shows the absolute difference map between the denoised projection and the ground truth.

    Args:
        denoised: Tensor of shape (C, H, W).
        ground_truth: Tensor of shape (C, H, W).
        filename: Output picture file.
        title: Title of the whole picture.

    Returns:
        List with the mean absolute difference of the 2 channel.
    """
    difference = (denoised - ground_truth).abs()

    # averages over the last two dimensions i.e. H and W of tensor of shape (C, H, W)
    mean_differences_tensor = difference.mean(dim=(1, 2))

    # converts that tensor to a Python list i.e. [mean_LE, mean_HE]
    mean_differences = mean_differences_tensor.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), squeeze=False)

    # --- Row 0: LE channel ---
    im00 = axes[0, 0].imshow(denoised[0].numpy(), cmap='gray')
    axes[0, 0].set_title("Denoised (LE)")
    fig.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im01 = axes[0, 1].imshow(ground_truth[0].numpy(), cmap='gray')
    axes[0, 1].set_title("Ground truth (LE)")
    fig.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im02 = axes[0, 2].imshow(difference[0].numpy(), cmap='gray')
    axes[0, 2].set_title(f"Absolute difference (LE), mean = {mean_differences[0]:.6f}")
    fig.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # --- Row 1: HE channel ---
    im10 = axes[1, 0].imshow(denoised[1].numpy(), cmap='gray')
    axes[1, 0].set_title("Denoised (HE)")
    fig.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im11 = axes[1, 1].imshow(ground_truth[1].numpy(), cmap='gray')
    axes[1, 1].set_title("Ground truth (HE)")
    fig.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im12 = axes[1, 2].imshow(difference[1].numpy(), cmap='gray')
    axes[1, 2].set_title(f"Absolute difference (HE), mean = {mean_differences[1]:.6f}")
    fig.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close(fig)

    return mean_differences


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
            -the validation loss stored in the checkpoint which act as a reference for the best validation loss so far.
    """

    # loads a checkpoint (weights_only=True: the checkpoint only holds tensors and numbers)
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    # take the parameters of the saved model from the dictionary 
    model_param = checkpoint['model_state_dict']
    optimizer_param = checkpoint['optimizer_state_dict']
    scheduler_param = checkpoint['scheduler_state_dict']
    checkpointed_epoch = checkpoint['checkpointed_epoch']
    checkpointed_val_loss = checkpoint.get('val_loss') # .get() is used because 'val_loss' is stored by **extra.

    # load those parameters into their corresponding objects.
    model.load_state_dict(model_param)
    optimizer.load_state_dict(optimizer_param)
    scheduler.load_state_dict(scheduler_param)

    return checkpointed_epoch, checkpointed_val_loss
