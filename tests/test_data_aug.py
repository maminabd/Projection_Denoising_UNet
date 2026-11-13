
import glob
import os
import random
import torch
from libs.data_augmentation import rotation_augmentation
from libs.dataset import load_as_tensor
import pytest

PROJECTIONS_DIR = os.path.join(os.path.dirname(__file__), "projections_for_testing")


def load_random_projection(folder):
    """
    Load a randomly chosen mouse projection of shape (2, 300, 300) 
    from the 'input' or 'target' folder.
    
    Args:
        folder (str): The folder name, either 'input' or 'target'.
    """
    
    files = sorted(glob.glob(os.path.join(PROJECTIONS_DIR, folder, "*.npy")))
    assert files, f"No .npy files found in {os.path.join(PROJECTIONS_DIR, folder)}"
    chosen_file = random.choice(files)

    return load_as_tensor(chosen_file)

# Rotation angles: k=1: 90° , k=2: 180° , k=3: 270° , k=4: 360°
@pytest.mark.parametrize("k", [1,2,3,4])
def test_rotation_returns_tensor(k):

    """
    GIVEN: mouse input and target projection
    WHEN: the rotation_augmentation is applied
    THEN: it returns two torch tensors
    """

    projections = load_random_projection("input_testing"), load_random_projection("target_testing")
    inp_aug, tgt_aug = rotation_augmentation(*projections, k)
    
    assert isinstance(inp_aug, torch.Tensor), f"Expected rotation_augmentation returns a tensor, got {type(inp_aug)}"
    assert isinstance(tgt_aug, torch.Tensor), f"Expected rotation_augmentation returns a tensor, got {type(tgt_aug)}"

@pytest.mark.parametrize("k", [1,2,3,4])
def test_rotation_output_is_float32(k):
    """
    GIVEN: mouse input and target projection, both float32
    WHEN: the rotation_augmentation is applied
    THEN: both outputs are float32 as well
    """

    projections = load_random_projection("input_testing"), load_random_projection("target_testing")
    inp_aug, tgt_aug = rotation_augmentation(*projections, k)
    assert inp_aug.dtype == torch.float32, f"Expected rotation_augmentation returns a float32, got {inp_aug.dtype}"
    assert tgt_aug.dtype == torch.float32, f"Expected rotation_augmentation returns a float32, got {tgt_aug.dtype}"

@pytest.mark.parametrize("k", [1,2,3,4])
def test_rotation_output_has_the_shape_of_the_input(k):
    """
    GIVEN: mouse input and target projection of shape (2, 300, 300)
    WHEN: the rotation_augmentation is applied
    THEN: both outputs keep the shape (2, 300, 300)
    """
    projections = load_random_projection("input_testing"), load_random_projection("target_testing")
    inp_aug, tgt_aug = rotation_augmentation(*projections, k)
    assert inp_aug.shape == projections[0].shape, \
        f"Expected rotation_augmentation returns a shape {tuple(projections[0].shape)}, got {tuple(inp_aug.shape)}"
    assert tgt_aug.shape == projections[1].shape, \
        f"Expected rotation_augmentation returns a shape {tuple(projections[1].shape)}, got {tuple(tgt_aug.shape)}"

@pytest.mark.parametrize("k", [1,2,3,4])
def test_output_stays_in_the_intensity_range_of_the_input(k):
    """
    GIVEN: mouse input and target projection
    WHEN: the rotation_augmentation is applied
    THEN: After rotation output values lies in the data range of the original projections
    """
    projections = load_random_projection("input_testing"), load_random_projection("target_testing")
    inp_aug, tgt_aug = rotation_augmentation(*projections, k)
    
    assert torch.isclose(inp_aug.min(), projections[0].min(), atol=1e-6), \
        f"After rotation the minimum changed from {projections[0].min().item()} to {inp_aug.min().item()}"

    assert torch.isclose(inp_aug.max(), projections[0].max(), atol=1e-6), \
        f"After rotation the maximum changed from {projections[0].max().item()} to {inp_aug.max().item()}"

    assert torch.isclose(tgt_aug.min(), projections[1].min(), atol=1e-6), \
        f"After rotation the minimum changed from {projections[1].min().item()} to {tgt_aug.min().item()}"
    
    assert torch.isclose(tgt_aug.max(), projections[1].max(), atol=1e-6), \
        f"After rotation the maximum changed from {projections[1].max().item()} to {tgt_aug.max().item()}"
