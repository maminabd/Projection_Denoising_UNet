
import glob
import os
import random
import torch
from libs.data_augmentation import rotation_augmentation, flipping_augmentation
from libs.dataset import load_as_tensor, find_paired_files
import pytest
import numpy as np

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

ALL_PATH_PAIRS = list(zip(input_files, target_files))

# define the rotation angles parameter for the tests
ROTATION_KS = [1, 2, 3] # Rotation angles: k=1: 90° , k=2: 180° , k=3: 270°

# define the flip types parameter for the tests
FLIP_TYPES = [0, 1, 2] # Flip types: 0: horizontal , 1: vertical , 2: both


# ----------------------------------------------------
# tests for rotation_augmentation function
# ----------------------------------------------------

@pytest.mark.parametrize("k", ROTATION_KS)
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

@pytest.mark.parametrize("k", ROTATION_KS)
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

@pytest.mark.parametrize("k", ROTATION_KS)
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

@pytest.mark.parametrize("k", ROTATION_KS)
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

    assert torch.isclose(inp_aug.mean(), inp.mean(), atol=1e-6), \
            f"After rotation the mean changed from {inp.mean().item()} to {inp_aug.mean().item()}"

    assert torch.isclose(tgt_aug.mean(), tgt.mean(), atol=1e-6), \
                f"After rotation the mean changed from {tgt.mean().item()} to {tgt_aug.mean().item()}"
    

@pytest.mark.parametrize("k", ROTATION_KS)
@pytest.mark.parametrize("inp, tgt", ALL_PROJECTION_PAIRS)
def test_input_and_target_get_the_same_transformation(inp, tgt, k):
    """
    GIVEN: the same projections are used as input and as target
    WHEN: the rotation_augmentation is applied
    THEN: both rotated input and target are similar to each other, 
          so a pair is rotated in the same way
    """

    # the intesity values of inp and tgt are different (test would fail in any case)
    # to solve this, we use inp as tgt to just check if the same rotation is applied to both
    inp_aug, tgt_aug = rotation_augmentation(inp, inp, k)
    
    assert torch.equal(inp_aug, tgt_aug), "Input and target were rotated differently"


@pytest.mark.parametrize("k", ROTATION_KS)
def test_rotation_function_respects_the_channels_order(k):
    """
    GIVEN: an input and a target whose each channel is filled with a specific constant value
    WHEN: the rotation_augmentation is applied
    THEN: every channel still holds its own constant value, so the channels order is (LE and HE)
    """ 

    # Create input and target, each has shape (2, 300, 300)
    # Each channel is filled with a specific constant value
    inp = torch.stack([
        torch.full((300, 300), 1.0),
        torch.full((300, 300), 2.0),
    ])
    tgt = torch.stack([
        torch.full((300, 300), 3.0),
        torch.full((300, 300), 4.0),
    ])

    inp_aug, tgt_aug = rotation_augmentation(inp, tgt, k)

    # Check each channel of inp_aug is still filled with the same constant
    assert torch.allclose(inp_aug[0], torch.full((300, 300), 1.0)), \
        "inp channel 0 changed value after rotation"
    assert torch.allclose(inp_aug[1], torch.full((300, 300), 2.0)), \
        "inp channel 1 changed value after rotation"

    # Check each channel of tgt_aug is still filled with the same constant
    assert torch.allclose(tgt_aug[0], torch.full((300, 300), 3.0)), \
        "tgt channel 0 changed value after rotation"
    assert torch.allclose(tgt_aug[1], torch.full((300, 300), 4.0)), \
        "tgt channel 1 changed value after rotation"

# ----------------------------------------------------
# tests for flipping_augmentation function
# ----------------------------------------------------

@pytest.mark.parametrize("flip_type", FLIP_TYPES)
@pytest.mark.parametrize("inp, tgt", ALL_PROJECTION_PAIRS)
def test_flipping_returns_tensor(inp, tgt, flip_type):
    """
    GIVEN: mouse input and target projection
    WHEN: the flipping_augmentation is applied
    THEN: it returns two torch tensors
    """

    inp_aug, tgt_aug = flipping_augmentation(inp, tgt, flip_type)

    assert isinstance(inp_aug, torch.Tensor), f"Expected flipping_augmentation returns a tensor, got {type(inp_aug)}"
    assert isinstance(tgt_aug, torch.Tensor), f"Expected flipping_augmentation returns a tensor, got {type(tgt_aug)}"

@pytest.mark.parametrize("flip_type", FLIP_TYPES)
@pytest.mark.parametrize("inp, tgt", ALL_PROJECTION_PAIRS)
def test_flipping_output_is_float32(inp, tgt, flip_type):
    """
    GIVEN: mouse input and target projection, both float32
    WHEN: the flipping_augmentation is applied
    THEN: both outputs are float32 as well
    """

    inp_aug, tgt_aug = flipping_augmentation(inp, tgt, flip_type)
    assert inp_aug.dtype == torch.float32, f"Expected flipping_augmentation returns a float32, got {inp_aug.dtype}"
    assert tgt_aug.dtype == torch.float32, f"Expected flipping_augmentation returns a float32, got {tgt_aug.dtype}"

@pytest.mark.parametrize("flip_type", FLIP_TYPES)
@pytest.mark.parametrize("inp, tgt", ALL_PROJECTION_PAIRS)
def test_flipping_output_has_the_shape_of_the_input(inp, tgt, flip_type):
    """
    GIVEN: mouse input and target projection of shape (2, 300, 300)
    WHEN: the flipping_augmentation is applied
    THEN: both outputs keep the shape (2, 300, 300)
    """

    inp_aug, tgt_aug = flipping_augmentation(inp, tgt, flip_type)
    assert inp_aug.shape == inp.shape, \
        f"Expected flipping_augmentation returns a shape {tuple(inp.shape)}, got {tuple(inp_aug.shape)}"
    assert tgt_aug.shape == tgt.shape, \
        f"Expected flipping_augmentation returns a shape {tuple(tgt.shape)}, got {tuple(tgt_aug.shape)}"

@pytest.mark.parametrize("flip_type", FLIP_TYPES)
@pytest.mark.parametrize("inp, tgt", ALL_PROJECTION_PAIRS)
def test_flipping_output_stays_in_the_intensity_range_of_the_input(inp, tgt, flip_type):
    """
    GIVEN: mouse input and target projection
    WHEN: the flipping_augmentation is applied
    THEN: After flipping output values lies in the data range of the original projections,
          and the minimum, maximum and mean intensity are the same as the original ones
          (flipping only moves the pixels)
    """

    inp_aug, tgt_aug = flipping_augmentation(inp, tgt, flip_type)

    assert inp_aug.min() == inp.min(), \
        f"After flipping the minimum changed from {inp.min().item()} to {inp_aug.min().item()}"

    assert inp_aug.max() == inp.max(), \
        f"After flipping the maximum changed from {inp.max().item()} to {inp_aug.max().item()}"

    assert tgt_aug.min() == tgt.min(), \
        f"After flipping the minimum changed from {tgt.min().item()} to {tgt_aug.min().item()}"

    assert tgt_aug.max() == tgt.max(), \
        f"After flipping the maximum changed from {tgt.max().item()} to {tgt_aug.max().item()}"

    assert torch.isclose(inp_aug.mean(), inp.mean(), atol=1e-6), \
        f"After flipping the mean changed from {inp.mean().item()} to {inp_aug.mean().item()}"

    assert torch.isclose(tgt_aug.mean(), tgt.mean(), atol=1e-6), \
        f"After flipping the mean changed from {tgt.mean().item()} to {tgt_aug.mean().item()}"

@pytest.mark.parametrize("flip_type", FLIP_TYPES)
@pytest.mark.parametrize("inp, tgt", ALL_PROJECTION_PAIRS)
def test_flipping_input_and_target_get_the_same_transformation(inp, tgt, flip_type):
    """
    GIVEN: the same projections are used as input and as target
    WHEN: the flipping_augmentation is applied
    THEN: both outputs (input and target) are identical, so a pair are flipped in the same way
    """

    # the intesity values of inp and tgt are different (test would fail)
    # so we use inp as tgt as well to just check if the same flip is applied to both
    inp_aug, tgt_aug = flipping_augmentation(inp, inp, flip_type)

    assert torch.equal(inp_aug, tgt_aug), "Input and target were flipped differently"

@pytest.mark.parametrize("flip_type", FLIP_TYPES)
def test_flipping_respects_the_channels_order(flip_type):
    """
    GIVEN: an input and a target whose channels are each filled with a distinct constant value
    WHEN: the flipping_augmentation is applied
    THEN: every channel still holds its own constant value, so the channels order is (LE and HE)
    """

    inp = torch.stack([
        torch.full((300, 300), 1.0),
        torch.full((300, 300), 2.0),
    ])
    tgt = torch.stack([
        torch.full((300, 300), 3.0),
        torch.full((300, 300), 4.0),
    ])

    inp_aug, tgt_aug = flipping_augmentation(inp, tgt, flip_type)

    assert torch.allclose(inp_aug[0], torch.full((300, 300), 1.0)), \
        "inp channel 0 changed value after flipping"
    assert torch.allclose(inp_aug[1], torch.full((300, 300), 2.0)), \
        "inp channel 1 changed value after flipping"
    assert torch.allclose(tgt_aug[0], torch.full((300, 300), 3.0)), \
        "tgt channel 0 changed value after flipping"
    assert torch.allclose(tgt_aug[1], torch.full((300, 300), 4.0)), \
        "tgt channel 1 changed value after flipping"


@pytest.mark.parametrize("inp, tgt", ALL_PROJECTION_PAIRS)
def test_rotation_no_loss_of_body_texture(inp, tgt, k=1):
    """
    GIVEN: the mouse projections are used as input and as target
    WHEN: the rotation_augmentation is applied
    THEN: the projection after rotating 360 degrees (4 times 90 degrees) 
          is identical to the original projection,

    Further explanation:

        after rotating projections 4 times 90 degrees stepwise, if any part of the mouse body goes outside in any 90 degrees rotation,
        that lost region cannot be recovered anymore, and after 4 times 90 degrees rotation, when it get to the initial form, 
        there shouldn't be any difference between the original and the final rotated projection.
    """

    number_rotate = 4 # 360/90 = 4 times rotation
    inp_original = inp
    tgt_original = tgt
    for _ in range(number_rotate):
        inp_aug, tgt_aug = rotation_augmentation(inp, tgt, 1) # rotate 90 degrees each time
        inp = inp_aug
        tgt = tgt_aug

    assert torch.equal(inp, inp_original), f"there is loss of body texture in input after rotating the projection"
    assert torch.equal(tgt, tgt_original), f"there is loss of body texture in target after rotating the projection"     