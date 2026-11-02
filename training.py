
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import (check_files, find_paired_files, split_files, collate_batch)
import torch.nn.functional as F
from unet import UNet
from utils import (save_checkpoint, get_device, plot_history)
import argparse

# hyperparameters and data paths
VAL_SPLIT = 0.2 # fraction of the files used for validation
SEED = 42
NUM_WORKERS = 4 # number of subprocesses used to load and preprocess data in parallel with training
LEARNING_RATE = 1e-4
T_0, T_MULT, ETA_MIN = 100, 2, 1e-7  # hyperparameters of CosineAnnealingWarmRestarts scheduler
IMAGE_SHAPE = (300, 300, 2)  # (H, W, C) for each single projection

def parse_args():
    """Command-line arguments: the options that change from one run to the next."""
    parser = argparse.ArgumentParser(description="Train the U-Net projection denoiser")
    parser.add_argument('--input', required=True, help="Directory with the noisy input .npy files")
    parser.add_argument('--target', required=True, help="Directory with the clean target .npy files")
    parser.add_argument('--epochs', type=int, default=25, help="Number of training epochs")
    parser.add_argument('--batch', type=int, default=8, help="Batch size")
    parser.add_argument('--save_model_every', type=int, default=5, help="Save a checkpoint every N epochs")
    return parser.parse_args()
               


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
    
    args = parse_args()

    device = get_device()

    log_file = "log.txt"
    plot_file = "loss_history.png"

    # Data processing
    input_files, target_files = find_paired_files(args.input, args.target)

    # check if the datasets are acceptable
    check_files(input_files, target_files)

    # split the datasets in to training and validation sets.
    train_in, train_tg, val_in, val_tg = split_files(input_files, target_files, VAL_SPLIT, SEED)

    # create data loader for training and validation dataset with a batch_size
    train_loader = make_loader(train_in, train_tg, args.batch, shuffle=True, device=device)
    val_loader = make_loader(val_in, val_tg, args.batch, shuffle=False, device=device)

    # Model, optimizer, scheduler
    model = UNet(in_channels=2, out_channels=2).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_MULT, eta_min=ETA_MIN)

    # Training loop
    train_history, val_history, lr_history = [], [], []
    header_message = (
    "\nHyperparameters and training settings:\n"

    f"Validation split fraction: {VAL_SPLIT}\n"
    f"  Number of Training projections: {len(train_in)}\n" 
    f"  Number of validation projections: {len(val_in)}\n"
    f"Batch size: {args.batch}, number of epochs: {args.epochs}\n"
    f"Save the model after each: {args.save_model_every} epochs\n"
    f"Scheduler: CosineAnnealingWarmRestarts. Parameters:\n"
    f"  initial lr: {LEARNING_RATE}, \n"
    f"  T_0: {T_0}, \n"
    f"  T_mult: {T_MULT}, \n"
    f"  eta_min: {ETA_MIN}\n"

    "\nStart Training ...\n"
    )
    
    print(header_message, flush=True)
    # recreate the log file and write the header message to it
    print(header_message, file=open(log_file, mode='w', encoding='utf-8'))
    for epoch in range(0, args.epochs):

        train_losses = run_epoch(model, train_loader, device, optimizer)
        val_losses = run_epoch(model, val_loader, device) # No optimization during validating.

        # Update the learning rate scheduler
        scheduler.step()
        
        # Extract the updated learning rate from the optimizer for tracking history
        current_lr = optimizer.param_groups[0]['lr']
        
        # save the histories for plotting 
        train_history.append(train_losses)
        val_history.append(val_losses)
        lr_history.append(current_lr)

        if (epoch+1) % args.save_model_every == 0:
            save_checkpoint(f"UNet_epoch_{epoch+1:03d}.pth", model, optimizer, scheduler, epoch, train_loss=train_losses, val_loss=val_losses)

        message = (f"\n Epoch {epoch+1}/{args.epochs}:"
                   f"  Train loss: {train_losses:.6f}, validation loss: {val_losses:.6f}\n")

        print(message, flush=True)
        # append the message to the log file
        print(message, file=open(log_file, mode='a', encoding='utf-8'))
    
    # once the training finished, save the plot from trends of training loss and validation loss values wrt epochs and ...
    # plot the learning rate schedule wrt epochs
    plot_history(train_history, val_history, lr_history, plot_file)

    ending_message = (
        f"\nTraining finished after {args.epochs} epochs.\n"
        # print the last training and validation loss values from the loss history lists
        f"Final training loss: {train_history[-1]:.6f}, final validation loss: {val_history[-1]:.6f}\n"
        f"Training history plot saved to '{plot_file}'\n"
    )
    print(ending_message, flush=True)
    # append the ending message to the log file
    print(ending_message, file=open(log_file, mode='a', encoding='utf-8'))


if __name__ == '__main__':
    main()
