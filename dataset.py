
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch

IMAGE_SHAPE = (300, 300, 2)  # (H, W, C) for each single projection


def check_files(input_files, target_files, image_shape=IMAGE_SHAPE):

    """
    Check two things:

    1st: If the input data and output data have the same length (number of projections)

    2nd: If every input/target file has the expected shape.


    Args:
        input_files: List of input (noisy) .npy file paths.
        target_files: List of target (noiseless) .npy file paths, in the same order as inputs.
        image_shape: Expected shape (H, W, C) of every projection.

    Raises:
        ValueError: if the two lists have different lengths, or if a file whether in
            input or target does not have the expected shape i.e. (300, 300, 2)
    """

    if len(input_files) != len(target_files):
        raise ValueError("Number of input and target files differ!")

    for path in list(input_files):
        loaded_array = np.load(path)
        array_shape = loaded_array.shape
        if array_shape != tuple(image_shape):
            raise ValueError(f"{path}: expected shape {tuple(image_shape)}, got {array_shape}")

    for path in list(target_files):
        loaded_array = np.load(path)
        array_shape = loaded_array.shape
        if array_shape != tuple(image_shape):
            raise ValueError(f"{path}: expected shape {tuple(image_shape)}, got {array_shape}")


def find_paired_files(input_dir, target_dir, pattern="*.npy"):

    """
    List the files in input_dir and target_dir and check that they match by name.

    Args:
        input_dir: Directory with input (noisy) files.
        target_dir: Directory with target (clean) files.
        pattern: Glob pattern of the files to use.

    Returns:
        sorted paired file paths of input_files, target_files
    """


    # sort the path of .npy files of input and target.
    input_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    target_files = sorted(glob.glob(os.path.join(target_dir, pattern)))

    print(f"Found {len(input_files)} input files")
    print(f"Found {len(target_files)} target files")

    if len(input_files) == 0:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {input_dir}")

    if len(target_files) == 0:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {target_dir}")

    if len(input_files) != len(target_files):
        raise ValueError("Number of input and target files differ!")

    # extract the main name of .npy files of input and target
    input_names = [os.path.basename(name) for name in input_files]
    target_names = [os.path.basename(name) for name in target_files]

    if input_names != target_names:
        raise ValueError("Input and target files don't match by name!")

    return input_files, target_files


def split_files(input_files, target_files, validation_split=0.2, seed=42):

    """
    Split paired input and target dataset into training and validation sets.
    Args:
        input_files: List of input file paths.
        target_files: List of target file paths.
        validation_split: Fraction of files to use for validation. (1 - validation_split is for training)
        seed: Random seed for reproducibility.

    Returns:
        following lists: train_input, train_target, val_input, val_target
    """

    # using a function of sklearn for splitting
    train_pairs, val_pairs = train_test_split(
        list(zip(input_files, target_files)),
        test_size=validation_split,
        random_state=seed
    )

    # unzip the pairs into separate input and target lists
    train_input, train_target = map(list, zip(*train_pairs))
    val_input, val_target = map(list, zip(*val_pairs))

    return train_input, train_target, val_input, val_target


def load_as_tensor(path):
    
    """
    A helper function for "collate_batch", it loads .npy file and convert it into a PyTorch tensor. 

    Args:
        path: Path to a .npy file containing an (H, W, C) array.

    Returns:
        A (C, H, W) float32 torch.Tensor.
    """
    
    array_data = np.load(path).astype(np.float32)
    tensor_data = torch.from_numpy(array_data).permute(2, 0, 1)
    # (H, W, C) -> (C, H, W), PyTorch expects the format that channel is first index

    return tensor_data


def collate_batch(path_pairs):

    """
    Read a batch of projection pairs in tensor and stack them together.

    This is the collate function (collate_fn) of the DataLoader.
    The DataLoader selects the (input_path, target_path) pairs of the batch 
    and gives them to this function, which reads the files.

    Args:
        path_pairs: List of (input_path, target_path) tuples.

    Returns:
        inputs, targets: Tensors of shape (B, C, H, W).
    """


    inputs = torch.stack([load_as_tensor(input_path) for input_path, _ in path_pairs])
    targets = torch.stack([load_as_tensor(target_path) for _, target_path in path_pairs])
    return inputs, targets

