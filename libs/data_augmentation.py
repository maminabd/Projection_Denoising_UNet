
import torch

def rotation_augmentation(input_tensor, target_tensor, k):
    """
    Apply rotation augmentation using torch.rot90
    Rotation angles: 90°, 180°, 270°

    Args:
        input_tensor: Input tensor (C=2, H=300, W=300) - LE and HE channels 
        target_tensor: Target tensor (C=2, H=300, W=300) - LE and HE channels
        k: Number of times to rotate 90° counter-clockwise (1, 2, or 3)
            k>0 is equivalent to rotating 90° k times, counter-clockwise. 
            k<0 is equivalent to rotating 90° |k| times, clockwise.
            Base k values: 1 (90°), 2 (180°), 3 (270°)
            
    Returns:
        Rotated (input, target) tensors
    """

    # torch.rot90 rotates counter-clockwise in the plane specified by dims (H, W axes = [1, 2])
    rotated_input = torch.rot90(input_tensor, k, dims=[1, 2])
    rotated_target = torch.rot90(target_tensor, k, dims=[1, 2])

    return rotated_input, rotated_target

def flipping_augmentation(input_tensor, target_tensor, flip_type):
    """
    Flip the input and the target both in the same way.
    The flipping is horizontal, vertical, or both, depending on flip_type.

    Args:
        input_tensor: Input tensor (C, H, W).
        target_tensor: Target tensor (C, H, W).
        flip_type: Type of flip to apply. 0 for horizontal, 1 for vertical, 2 for both.

    Returns:
        Flipped (input, target) tensors, with the same shape as the inputs.
    """

    if flip_type == 0:
        dims = [2]      # horizontal flip (left-right): along the width
    elif flip_type == 1:
        dims = [1]      # vertical flip (up-down): along the height
    else:
        dims = [1, 2]   # both flips

    flipped_inp_tens, flipped_tg_tens = torch.flip(input_tensor, dims=dims), torch.flip(target_tensor, dims=dims)

    return flipped_inp_tens, flipped_tg_tens
