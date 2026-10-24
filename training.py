
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import (check_files, find_paired_files, split_files, collate_batch)
import torch.nn.functional as F
from unet import UNet

# hyperparameters and data paths
input_path = "./input"
target_path = "./target"
epochs = 4
save_model_every = 2
batch_size = 8
VAL_SPLIT = 0.2 # fraction of the files used for validation
SEED = 42
NUM_WORKERS = 4 # number of subprocesses used to load and preprocess data in parallel with training
LEARNING_RATE = 1e-4
T_0, T_MULT, ETA_MIN = 100, 2, 1e-7  # hyperparameters of CosineAnnealingWarmRestarts scheduler

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

def make_loader(input_files, target_files, batch_size, shuffle, device):
    """
    Build a DataLoader over the (input, target) file pairs.

    The DataLoader works directly on the list of path pairs and reads the
    files through collate_batch

    Args:
        input_files, target_files: lists of paired file paths.
        batch_size: number of projection files per batch.
        shuffle: whether to shuffle the pairs at every epoch.
        device: torch device.

    Returns:
        The DataLoader.
    """
    path_pairs = list(zip(input_files, target_files))

    dataloader = DataLoader(path_pairs, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_batch, num_workers=NUM_WORKERS,
                      pin_memory=device.type == 'cuda')

    return dataloader


def run_epoch(model, loader, device, optimizer=None):

    """
    Run training or validation over whole datasets each time it is called.

    Args:
        model: the U-Net.
        loader: DataLoader returning (inputs, targets), each of shape (B, 2, H, W).
        device: torch device.
        optimizer: optimizer of model's weights. If None, the epoch is a validation epoch and no
            gradient is computed, otherwise it is training epoch.

    Returns:
        Dictionary with the loss.
    """

    training = optimizer is not None
    model.train(training)
    np.random.seed(SEED)
    total_loss = 0.0
    n_batches = 0

    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = F.l1_loss(outputs, targets) # loss function is mean absolute error

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def main():

    device = torch.device('cuda')

    log_file = "log.txt"
    plot_file = "loss_history.png"

    # Data processing
    input_files, target_files = find_paired_files(input_path, target_path)

    # check if the datasets are acceptable
    check_files(input_files, target_files)

    # split the datasets in to training and validation sets.
    train_in, train_tg, val_in, val_tg = split_files(input_files, target_files, VAL_SPLIT, SEED)

    # create data loader for training and validation dataset with a batch_size
    train_loader = make_loader(train_in, train_tg, batch_size, shuffle=True, device=device)
    val_loader = make_loader(val_in, val_tg, batch_size, shuffle=False, device=device)

    print(f"Training on {len(train_in)} projections, validating on {len(val_in)}")

    # Model, optimizer, scheduler
    model = UNet(in_channels=2, out_channels=2).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_MULT, eta_min=ETA_MIN)

    # Training loop
    train_history, val_history = [], []
    print("Start Training ... ", file=open(log_file, 'w', encoding='utf-8'))
    for epoch in range(0, epochs):

        train_losses = run_epoch(model, train_loader, device, optimizer)
        val_losses = run_epoch(model, val_loader, device) # No optimization during validating.

        scheduler.step()

        train_history.append(train_losses)
        val_history.append(val_losses)

        if (epoch+1) % save_model_every == 0:
            save_checkpoint(f"UNet_epoch_{epoch+1:03d}.pth", model, optimizer, scheduler, epoch, train_loss=train_losses, val_loss=val_losses)

        message = (f"\n Epoch {epoch+1}/{epochs}:"
                   f"  Train loss: {train_losses:.6f}, validation loss: {val_losses:.6f}\n")

        print(message, flush=True)
        print(message, file=open(log_file, 'a', encoding='utf-8'))

    print("\n\n Training completed \n\n")


if __name__ == '__main__':
    main()
