import argparse
import logging
import os
import random
import time
from functools import partial

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from libs.dataset import (check_files, find_paired_files, split_files, 
                          collate_batch, collate_augmented_batch)
from libs.losses import denoising_loss
from libs.unet import UNet
from libs.utils import (save_checkpoint, get_device, plot_history, load_checkpoint)

logger = logging.getLogger(__name__)

# hyperparameters and data paths
VAL_SPLIT = 0.2 # fraction of the files used for validation
SEED = 42
NUM_WORKERS = 4 # number of subprocesses used to load and preprocess data in parallel with training
LEARNING_RATE = 1e-4
T_0, T_MULT, ETA_MIN = 50, 2, 1e-7  # hyperparameters of CosineAnnealingWarmRestarts scheduler
MAE_WEIGHT, SSIM_WEIGHT, GRADIENT_WEIGHT = 0.5, 0.25, 0.25

# Data Augmentation Configuration
USE_AUGMENTATION = True  # Enable/disable rotation and flip augmentation
ROTATION_PROBABILITY = 0.25  # Probability of applying rotation to each training sample
FLIP_PROBABILITY = 0.25 # Probability of applying horizontal and/or vertical flip to each training sample

def parse_args():
    """Command-line arguments: the options that change from one run to the next."""
    parser = argparse.ArgumentParser(description="Train the U-Net projection denoiser")
    parser.add_argument('--dataset', required=True, help="Directory to the dataset folder containing 'input' and 'target' subfolders. Default: None (required)")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs. Default: 100")
    parser.add_argument('--batch', type=int, default=8, help="Batch size. Default: 8")
    parser.add_argument('--save_model_every', type=int, default=25, help="Save a checkpoint every N epochs. Default: 25")
    parser.add_argument('--plot_every', type=int, default=5, help="Plot training history every N epochs. Default: 5")
    parser.add_argument('--resume', default=None, help="Path to checkpoint (.pth) to continue training from. Default: None (start training from scratch)")
    parser.add_argument('--check_dataset', default=None, help="Check if the dataset is valid and has the expected shape and " \
                                                            "intensity range before training. For the first run, it is " \
                                                            "recommended to check the dataset first (e.g. --check_dataset 1). " \
                                                            "If the dataset is valid, this option can be omitted in " \
                                                            "subsequent runs. Default: None (do not check the dataset)")
                                                                                                    
    return parser.parse_args()
               
def make_loader(input_files, target_files, batch_size, shuffle, device, use_augmentation=False):
    """
    Build a DataLoader over the (input, target) file pairs.

    The DataLoader works directly on the list of path pairs and reads the
    files through collate_batch (or collate_augmented_batch if use_augmentation is True)

    Args:
        input_files, target_files: lists of paired file paths.
        batch_size: number of projection files per batch.
        shuffle: whether to shuffle the pairs at every epoch.
        device: torch device.
        use_augmentation: whether to augment (rotate/flip) the projections by a probability 
                          (for the training set only).

    Returns:
        The DataLoader.
    """
    path_pairs = list(zip(input_files, target_files))

    # partial (instead of a lambda) keeps the collate function picklable, 
    # which is necessary for the DataLoader workers on the platforms that start them with "spawn" (Windows, macOS)
    if use_augmentation:
        collate_fn = partial(collate_augmented_batch, use_augmentation=True,
                             rotation_probability=ROTATION_PROBABILITY, flip_probability=FLIP_PROBABILITY)
    else:
        collate_fn = collate_batch

    dataloader = DataLoader(path_pairs, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                            num_workers=NUM_WORKERS, pin_memory=device.type == 'cuda')

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
        The average loss over the batches of the epoch.
    """

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    n_batches = 0

    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            
            # multi-component loss
            loss = denoising_loss(outputs, targets, MAE_WEIGHT, SSIM_WEIGHT, GRADIENT_WEIGHT)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def setup_logging(log_file, append):
    """
    Send the log messages both to the terminal and to the log file.

    Args:
        log_file: path of the log file.
        append: if True the messages are appended to the existing log file (resumed training),
                otherwise the log file is recreated.
    """

    formatter = logging.Formatter("%(message)s") # only the message, no timestamp, no level, no module name

    # handlers send the log messages to both the terminal and the log file
    handlers = [logging.StreamHandler(),
                logging.FileHandler(log_file, mode='a' if append else 'w', encoding='utf-8')]

    root_logger = logging.getLogger() # get the root logger, which is the parent of all loggers in the hierarchy
    root_logger.setLevel(logging.INFO) # set the logging level to INFO, so that all messages of level INFO and above are processed

    # add the handlers to the root logger, so that all log messages are sent to both the terminal and the log file
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def main():
    
    args = parse_args()

    log_file = "log.txt"
    plot_file = "loss_history.png"

    # a resumed training continues the log of the previous run instead of replacing it
    setup_logging(log_file, append=bool(args.resume))

    # fix the random seeds, to make the training replicable
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()

    # get input/target directories from the fixed subfolder names,
    input_dir = os.path.join(args.dataset, 'input')
    target_dir = os.path.join(args.dataset, 'target')

    # Data processing
    input_files, target_files = find_paired_files(input_dir, target_dir)

    # check if the training dataset is acceptable.
    if args.check_dataset:
        check_files(input_files, target_files)
        logger.info(f"\nDataset in '{args.dataset}' is valid and has the expected shape and intensity range.")

    # split the datasets in to training and validation sets.
    train_in, train_tg, val_in, val_tg = split_files(input_files, target_files, VAL_SPLIT, SEED)

    # create data loader for training and validation dataset with a batch_size
    train_loader = make_loader(train_in, train_tg, args.batch, shuffle=True, device=device, use_augmentation=USE_AUGMENTATION)
    # data augmentation is applied only to the training dataset, not to the validation dataset.
    val_loader = make_loader(val_in, val_tg, args.batch, shuffle=False, device=device)

    # Model, optimizer, scheduler
    model = UNet(in_channels=2, out_channels=2).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_MULT, eta_min=ETA_MIN)
    
    start_epoch = 0
    best_val_loss = float('inf')
    # Resume training from a checkpoint if provided
    if args.resume:
        if os.path.exists(args.resume):
            checkpointed_epoch, checkpointed_val_loss = load_checkpoint(args.resume, model, optimizer, scheduler, device)
            
            start_epoch = checkpointed_epoch # load_checkpoint already returns the epoch after the checkpointed one.
            best_val_loss = checkpointed_val_loss # the last validation loss stored in the checkpoint is used as a reference for the best validation loss so far.

            if start_epoch >= args.epochs:
                logger.warning(f"Specified number of epochs ({args.epochs}) is less than or equal to the checkpointed epoch ({start_epoch}). "
                               f"No training will be performed. Increase the number of epochs.")
                return
            
            logger.info(f"\n\nContinuing from checkpoint {args.resume}, "
                        f"at epoch {start_epoch}, best validation loss {best_val_loss:.6f}")
        
        else:
            raise FileNotFoundError(f"Checkpoint '{args.resume}' not found")

                                                                                                      
    # Training loop
    train_history, val_history, lr_history = [], [], []
    header_message = (
    "\nHyperparameters and training settings:\n\n"
    f"Validation split fraction: {VAL_SPLIT}\n"
    f"  Number of Training projections: {len(train_in)}\n" 
    f"  Number of validation projections: {len(val_in)}\n"
    f"Batch size: {args.batch}, number of epochs: {args.epochs}\n"
    f"Save the model after each: {args.save_model_every} epochs\n"
    f"Data augmentation: {USE_AUGMENTATION} (rotation probability {ROTATION_PROBABILITY}, "
    f"flip probability {FLIP_PROBABILITY})\n"
    f"Scheduler: CosineAnnealingWarmRestarts. Parameters:\n"
    f"  initial lr: {LEARNING_RATE}, \n"
    f"  T_0: {T_0}, \n"
    f"  T_mult: {T_MULT}, \n"
    f"  eta_min: {ETA_MIN}\n"

    "\nStart Training ...\n"
    )
    
    logger.info(header_message)

    for epoch in range(start_epoch, args.epochs):

        start_time = time.time()
        
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device) # No optimization during validating.

        # Update the learning rate scheduler
        scheduler.step()
        
        # Extract the updated learning rate from the optimizer for tracking history
        current_lr = optimizer.param_groups[0]['lr']
        
        # save the histories for plotting 
        train_history.append(train_loss)
        val_history.append(val_loss)
        lr_history.append(current_lr)

        logger.info(f"\nEpoch {epoch+1}/{args.epochs}, ({time.time() - start_time:.1f} seconds):"
                    f"  Train loss: {train_loss:.6f}, validation loss: {val_loss:.6f}, learning rate: {current_lr:.6e}\n")

        # The best model is the one with the lowest validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint("UNet_best.pth", model, optimizer, scheduler, epoch,
                            train_loss=train_loss, val_loss=val_loss)
            
            logger.info(f"checkpoint saved at epoch {epoch+1} with best validation loss: {best_val_loss:.6f} so far\n")

        if (epoch+1) % args.save_model_every == 0:
            save_checkpoint(f"UNet_epoch_{epoch+1:03d}.pth", model, optimizer, scheduler, epoch, train_loss=train_loss, val_loss=val_loss)
                    
            logger.info(f"Checkpoint saved at epoch {epoch+1}\n")

        if (epoch + 1) % args.plot_every == 0:
            plot_history(train_history, val_history, lr_history, start_epoch + 1, plot_file)

    # ------------------------------------------------------------------
    # End of training
    # ------------------------------------------------------------------
    
    # once the training finished, save the plot from trends of training loss and validation loss values wrt epochs and ...
    # plot the learning rate schedule wrt epochs
    if train_history:
        plot_history(train_history, val_history, lr_history, start_epoch + 1, plot_file)

    logger.info(
        f"\nTraining finished after {args.epochs} epochs.\n"
        # print the last training and validation loss values from the loss history lists
        f"Final training loss: {train_history[-1]:.6f}, final validation loss: {val_history[-1]:.6f}\n"
        f"Training history plot saved to '{plot_file}'\n"
    )


if __name__ == '__main__':
    main()
