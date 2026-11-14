
import glob
import os
import random
import torch
from libs.data_augmentation import rotation_augmentation
from libs.dataset import load_as_tensor, find_paired_files
import pytest

# define the path to the projections_for_testing folder which contains 
# input_testing and target_testing subfolders of mouse projections for testing purposes
PROJECTIONS_DIR = os.path.join(os.path.dirname(__file__), "projections_for_testing")

# from the folders, find the paired input and target files for testing
input_files, target_files = find_paired_files(
    os.path.join(PROJECTIONS_DIR, "input_testing"),
    os.path.join(PROJECTIONS_DIR, "target_testing"),)

# load the paired input and target files as torch tensors
ALL_PROJECTION_PAIRS = [
    (load_as_tensor(inp), load_as_tensor(tgt))
    for inp, tgt in zip(input_files, target_files)]

# define the rotation angles parameter for the tests
ANGLES = [1, 2, 3] # Rotation angles: k=1: 90° , k=2: 180° , k=3: 270°

@pytest.mark.parametrize("k", ANGLES)
@pytest.mark.parametrize("inp, tgt", ALL_PROJECTION_PAIRS)
def test_rotation_returns_tensor(inp, tgt, k):

    """
    GIVEN: mouse input and target projection
    WHEN: the rotation_augmentation is applied
    THEN: it returns two torch tensors
    """

    inp_aug, tgt_aug = rotation_augmentation(inp, tgt, k)
    
    assert isinstance(inp_aug, torch.Tensor), f"Expected rotation_augmentation returns a tensor, got {type(inp_aug)}"
    assert isinstance(tgt_aug, torch.Tensor), f"Expected rotation_augmentation returns a tensor, got {type(tgt_aug)}"

@pytest.mark.parametrize("k", ANGLES)
@pytest.mark.parametrize("inp, tgt", ALL_PROJECTION_PAIRS)
def test_rotation_output_is_float32(inp, tgt, k):
    """
    GIVEN: mouse input and target projection, both float32
    WHEN: the rotation_augmentation is applied
    THEN: both outputs are float32 as well
    """

    inp_aug, tgt_aug = rotation_augmentation(inp, tgt, k)
    assert inp_aug.dtype == torch.float32, f"Expected rotation_augmentation returns a float32, got {inp_aug.dtype}"
    assert tgt_aug.dtype == torch.float32, f"Expected rotation_augmentation returns a float32, got {tgt_aug.dtype}"

@pytest.mark.parametrize("k", ANGLES)
@pytest.mark.parametrize("inp, tgt", ALL_PROJECTION_PAIRS)
def test_rotation_output_has_the_shape_of_the_input(inp, tgt, k):
    """
    GIVEN: mouse input and target projection of shape (2, 300, 300)
    WHEN: the rotation_augmentation is applied
    THEN: both outputs keep the shape (2, 300, 300)
    """
    
    inp_aug, tgt_aug = rotation_augmentation(inp, tgt, k)
    assert inp_aug.shape == inp.shape, \
        f"Expected rotation_augmentation returns a shape {tuple(inp.shape)}, got {tuple(inp_aug.shape)}"
    assert tgt_aug.shape == tgt.shape, \
        f"Expected rotation_augmentation returns a shape {tuple(tgt.shape)}, got {tuple(tgt_aug.shape)}"

@pytest.mark.parametrize("k", ANGLES)
@pytest.mark.parametrize("inp, tgt", ALL_PROJECTION_PAIRS)
def test_output_stays_in_the_intensity_range_of_the_input(inp, tgt, k):
    """
    GIVEN: mouse input and target projection
    WHEN: the rotation_augmentation is applied
    THEN: After rotation output values lies in the data range of the original projections
    """

    inp_aug, tgt_aug = rotation_augmentation(inp, tgt, k)
    
    assert torch.isclose(inp_aug.min(), inp.min(), atol=1e-6), \
        f"After rotation the minimum changed from {inp.min().item()} to {inp_aug.min().item()}"

    assert torch.isclose(inp_aug.max(), inp.max(), atol=1e-6), \
        f"After rotation the maximum changed from {inp.max().item()} to {inp_aug.max().item()}"

    assert torch.isclose(tgt_aug.min(), tgt.min(), atol=1e-6), \
        f"After rotation the minimum changed from {tgt.min().item()} to {tgt_aug.min().item()}"
    
    assert torch.isclose(tgt_aug.max(), tgt.max(), atol=1e-6), \
        f"After rotation the maximum changed from {tgt.max().item()} to {tgt_aug.max().item()}"
