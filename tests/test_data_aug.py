
import glob
import os
import random
import torch
from libs.data_augmentation import rotation_augmentation
from libs.dataset import load_as_tensor, find_paired_files, collate_augmented_batch, collate_batch
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
# tests for collate_augmented_batch function
# ----------------------------------------------------

@pytest.mark.parametrize("path_pairs", ALL_PATH_PAIRS)
def test_collate_augmented_batch_is_unchanged_when_rotation_is_disabled(path_pairs):
    """
    GIVEN: batch of mouse projections while rotation augmentation is disabled but probability is 1
    WHEN: collate_augmented_batch is called
    THEN: the batch returned by collate_augmented_batch equals
          the one returned by collate_batch (without augmentation).
    """

    path_pairs = [path_pairs]

    expected_inputs, expected_targets = collate_batch(path_pairs)

    inputs, targets = collate_augmented_batch(
        path_pairs,
        use_augmentation=False,
        rotation_probability=1.0,
        flip_probability=0.0
    )

    assert torch.equal(inputs, expected_inputs), \
        "The inputs changed although rotation augmentation is disabled"

    assert torch.equal(targets, expected_targets), \
        "The targets changed although rotation augmentation is disabled"

@pytest.mark.parametrize("path_pairs", ALL_PATH_PAIRS)
def test_collate_augmented_batch_is_unchanged_with_zero_probabilities(path_pairs):
    """
    GIVEN: batch of mouse projections while rotation probability is 0 for augmentation.
    WHEN: collate_augmented_batch is called.
    THEN: the batch returned by collate_augmented_batch equals
          the one returned by collate_batch (without augmentation).
    """
    
    path_pair = [path_pairs]

    expected_inputs, expected_targets = collate_batch(path_pair)

    inputs, targets = collate_augmented_batch(
        path_pair,
        use_augmentation=True,
        rotation_probability=0.0,
        flip_probability=0.0
    )

    assert torch.equal(inputs, expected_inputs), "The inputs changed although the rotation probability is 0"
    assert torch.equal(targets, expected_targets), "The targets changed although the rotation probability is 0"

@pytest.mark.parametrize("path_pair", ALL_PATH_PAIRS[:1]) 
# only test on the first projection pair to avoid long test time
def test_collate_augmented_batch_rotates_expected_numberof_projections(path_pair):
    """
    GIVEN: a batch of 100 copies of a mouse projection while rotation is enabled
           with a probability of 0.25 for augmentation
    WHEN: collate_augmented_batch is called with a seeded generator
    THEN: about 25% of the projections are rotated
    """

    path_pairs = [path_pair]*100  # create a batch of 100 copies of the same projection
    # does not matter which projection is used, we just want to check the share of rotated projections
    original_inputs, _ = collate_batch(path_pairs)

    # Seed the random generator to make the test reproducible
    random.seed(0)

    inputs, _ = collate_augmented_batch(
        path_pairs,
        use_augmentation=True,
        rotation_probability=0.25,
        flip_probability=0.0
    )

    n_rotated = 0
    for projection in inputs:
        if not torch.equal(projection, original_inputs[0]):
            n_rotated += 1

    assert abs(n_rotated / 100 - 0.25) < 0.1, f"Expected about 25% rotated projections, got {n_rotated}%"


@pytest.mark.parametrize("path_pair", ALL_PATH_PAIRS[:1]) 
# only test on the first pair to avoid long test time
def test_collate_augmented_batch_uses_every_rotation_angle(path_pair):

    """
    GIVEN: a batch of many copies of the same mouse projection while
           the rotation enabled with a probability of 1.
    WHEN: collate_augmented_batch is called with a seeded generator.
    THEN: every rotation (k=1, 2 and 3) occurs in the batch,
    """

    path_pair = [path_pair]
    original_inputs, _ = collate_batch(path_pair)

    # The 3 possible cases of rotating a (C, H, W) tensor, in the order of ANGLE_KS.
    rotation_case1 = torch.rot90(original_inputs[0], 1, dims=[1, 2])
    rotation_case2 = torch.rot90(original_inputs[0], 2, dims=[1, 2])
    rotation_case3 = torch.rot90(original_inputs[0], 3, dims=[1, 2])
    all_rotation_cases = [rotation_case1, rotation_case2, rotation_case3]
    
    random.seed(0)

    inputs, _ = collate_augmented_batch(
        path_pair*100,
        use_augmentation=True,
        rotation_probability=1.0,
        flip_probability=0.0
    )

    rotations_found = []
    for projection in inputs:
        for k in ROTATION_KS:
            if torch.equal(projection, all_rotation_cases[k-1]):
                if k not in rotations_found:
                    rotations_found.append(k)

    assert 1 in rotations_found, "Rotation k=1 was never found in the batch"
    assert 2 in rotations_found, "Rotation k=2 was never found in the batch"
    assert 3 in rotations_found, "Rotation k=3 was never found in the batch"

        
@pytest.mark.parametrize("path_pair", ALL_PATH_PAIRS[:1]) 
# only test on the first pair to avoid long test time
def test_collate_augmented_batch_rotation_distribution_is_uniform(path_pair):
    """
    GIVEN: a batch of many copies of the same mouse projection while
           the rotation enabled with a probability of 1.
    WHEN: collate_augmented_batch is called with a seeded generator
    THEN: each of the 3 possible rotations (k=1, 2, 3) occurs roughly 1/3 of the time,
          confirming the angle is chosen uniformly at random for each projection
    """

    path_pair = [path_pair]
    original_inputs, _ = collate_batch(path_pair)

    # The 3 possible cases of rotating a (C, H, W) tensor, in the order of ANGLE_KS.
    rotation_case1 = torch.rot90(original_inputs[0], 1, dims=[1, 2])
    rotation_case2 = torch.rot90(original_inputs[0], 2, dims=[1, 2])
    rotation_case3 = torch.rot90(original_inputs[0], 3, dims=[1, 2])
    all_rotation_cases = [rotation_case1, rotation_case2, rotation_case3]

    random.seed(0)

    batch_size = 300  # large enough for the counts to settle near the true probability
    inputs, _ = collate_augmented_batch(
            path_pair*batch_size,
            use_augmentation=True,
            rotation_probability=1.0,
            flip_probability=0.0
        ) # flipping is disabled to avoid interference with rotation counts
    
    # Count how many times each rotation k was found in the batch.
    rotations_found = {1: 0, 2: 0, 3: 0}
    for projection in inputs:
        for k in ROTATION_KS:
            if torch.equal(projection, all_rotation_cases[k-1]):
                rotations_found[k] += 1

    # Convert counts into proportions out of the whole batch.
    proportion = {}
    proportion[1] = rotations_found[1] / batch_size
    proportion[2] = rotations_found[2] / batch_size
    proportion[3] = rotations_found[3] / batch_size

    expected_proportion = 1 / len(ROTATION_KS)  # should be 1/3 for all k values
    tolerance = 0.03  # allow +/- 3 percentage points of random noise

    assert abs(proportion[1] - expected_proportion) < tolerance, (
                f"Rotation k=1 occurred {proportion[1]:.2%} of the time, "
                f"expected close to {expected_proportion:.2%}")
   
    assert abs(proportion[2] - expected_proportion) < tolerance, (
                f"Rotation k=2 occurred {proportion[2]:.2%} of the time, "
                f"expected close to {expected_proportion:.2%}")
    
    assert abs(proportion[3] - expected_proportion) < tolerance, (
                f"Rotation k=3 occurred {proportion[3]:.2%} of the time, "
                f"expected close to {expected_proportion:.2%}")
