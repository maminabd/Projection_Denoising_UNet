
import numpy as np

IMAGE_SHAPE = (300, 300, 2)  # (H, W, C) for each single projection


def check_files(input_files, target_files, image_shape=IMAGE_SHAPE):

    """
    Check two things:

    1st: If the input data and output data have the same length (number of projections)

    2nd: If every input/target file has the expected shape.


    Args:
        input_files: List of input (noisy) .npy file paths.
        target_files: List of target (clean) .npy file paths, in the same order.
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
