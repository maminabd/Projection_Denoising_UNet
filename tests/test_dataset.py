
import os
import numpy as np
import pytest
from libs.dataset import (check_files, find_paired_files, split_files, 
                          collate_batch, collate_augmented_batch, load_as_tensor)
import torch
import random

IMAGE_SHAPE = (300, 300, 2)  # (H, W, C) for each single projection. This is constant for this work.

# define the path to the projections_for_testing folder which contains 
# input_testing and target_testing subfolders of mouse projections for testing purposes                                                                                                                                                                                                                                                                                                               
PROJECTIONS_DIR = os.path.join(os.path.dirname(__file__), "projections_for_testing")

# from the folders, find the paired input and target files for testing                                                                                                                                            
input_files, target_files = find_paired_files(
    os.path.join(PROJECTIONS_DIR, "input_testing"),
    os.path.join(PROJECTIONS_DIR, "target_testing"),)

# create a list of touples from the paired input and target file paths
ALL_PATH_PAIRS = list(zip(input_files, target_files)) 

# define the rotation angles parameter for the tests
ROTATION_KS = [1, 2, 3] # Rotation angles: k=1: 90° , k=2: 180° , k=3: 270°

# define the flip types parameter for the tests
FLIP_TYPES = [0, 1, 2] # Flip types: 0: horizontal , 1: vertical , 2: both

def test_check_files_accepts_valid_files(tmp_path):

    """
    GIVEN: input and target dataset with equal length whose all npy files have the expected shape
    WHEN: check_files is called
    THEN: no error is raised
    """

    input_dir = tmp_path / "input_data"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_list = []
    for i in range(2):
        input_path = input_dir / f"mouse_{i}.npy"
        np.save(input_path, np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
        input_list.append(str(input_path))

    target_dir = tmp_path / "target_data"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_list = []
    for i in range(2):
        target_path = target_dir / f"mouse_{i}.npy"
        np.save(target_path, np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
        target_list.append(str(target_path))

    check_files(input_list, target_list)

def test_check_files_invalid_intensity_values_raises_for_inputs(tmp_path):

    """
    GIVEN: input and targets .npy files and
           intensity values of inputs are outside the range [0, 1]
    WHEN: check_files is called
    THEN: a ValueError is raised
    """
    torch.manual_seed(42)

    # create directories and save the .npy files
    input_dir = tmp_path / "input_data"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / "mouse_1.npy"
    input_path_str = str(input_path)
    np.save(input_path, np.full(IMAGE_SHAPE, 1.5, dtype=np.float32)) # values outside range [0, 1]

    target_dir = tmp_path / "target_data"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "mouse_1.npy"
    target_path_str = str(target_path)
    np.save(target_path, np.full(IMAGE_SHAPE, 0.5, dtype=np.float32)) # values inside range [0, 1]
      
    with pytest.raises(ValueError, match="intensity"):
        check_files([input_path_str], [target_path_str])

def test_check_files_invalid_intensity_values_raises_for_targets(tmp_path):

    """
    GIVEN: input and targets .npy files and
           intensity values of targets are outside the range [0, 1]
    WHEN: check_files is called
    THEN: a ValueError is raised
    """
    torch.manual_seed(42)

    # create directories and save the .npy files
    input_dir = tmp_path / "input_data"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / "mouse_1.npy"
    input_path_str = str(input_path)
    np.save(input_path, np.full(IMAGE_SHAPE, 0.5, dtype=np.float32)) # values inside range [0, 1]

    target_dir = tmp_path / "target_data"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "mouse_1.npy"
    target_path_str = str(target_path)
    np.save(target_path, np.full(IMAGE_SHAPE, 1.5, dtype=np.float32)) # values outside range [0, 1]
        
    with pytest.raises(ValueError, match="intensity"):
        check_files([input_path_str], [target_path_str])


def test_check_files_length_mismatch_raises(tmp_path):

    """
    GIVEN: input is 3 files but target is 2 files
    WHEN: check_files is called
    THEN: a ValueError is raised
    """

    # create directories and save the .npy files
    input_dir = tmp_path / "input_data"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_files = []
    for i in range(3):
        path = input_dir / f"mouse_{i}.npy"
        np.save(path, np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
        input_files.append(str(path))

    target_dir = tmp_path / "target_data"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_files = []
    for i in range(2):
        path = target_dir / f"mouse_{i}.npy"
        np.save(path, np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
        target_files.append(str(path))

    with pytest.raises(ValueError, match="differ"):
        check_files(input_files, target_files)


def test_check_files_wrong_shape_raises_for_inputs(tmp_path):

    """
    GIVEN: input and targets .npy files and
           shape of inputs are not the expected shape i.e. (300, 300, 2)
    WHEN: check_files is called
    THEN: a ValueError naming the expected shape is raised
    """

    # create directories and save the .npy files
    input_dir = tmp_path / "input_data"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / "mouse_1.npy"
    input_list = [str(input_path)]
    np.save(input_path, np.full((300, 300), 0.5, dtype=np.float32))  # wrong shape

    target_dir = tmp_path / "target_data"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "mouse_1.npy"
    target_list = [str(target_path)]
    np.save(target_path, np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))  # correct shape

    with pytest.raises(ValueError, match="expected shape"):
        check_files(input_list, target_list)

def test_check_files_wrong_shape_raises_for_targets(tmp_path):

    """
    GIVEN: input and targets .npy files and
           shape of targets are not the expected shape i.e. (300, 300, 2)
    WHEN: check_files is called
    THEN: a ValueError naming the expected shape is raised
    """

    # create directories and save the .npy files
    input_dir = tmp_path / "input_data"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / "mouse_1.npy"
    input_list = [str(input_path)]
    np.save(input_path, np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))  # wrong shape

    target_dir = tmp_path / "target_data"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "mouse_1.npy"
    target_list = [str(target_path)]
    np.save(target_path, np.full((300, 300), 0.5, dtype=np.float32))  # correct shape

    with pytest.raises(ValueError, match="expected shape"):
        check_files(input_list, target_list)

def test_find_paired_files_missing_npyfiles_raises_for_input(tmp_path):

    """
    GIVEN: input directory with no file that matching the pattern of *.npy
    WHEN: find_paired_files is called
    THEN: a FileNotFoundError is raised
    """

    # create directories
    input_dir =  tmp_path / "input"
    target_dir = tmp_path / "target"

    input_dir.mkdir()
    target_dir.mkdir()

    # save the .npy files in the directories
    # input directory has no .npy files, but has a .txt file
    (input_dir / "mouse.txt").write_text("hello world\n")

    # target directory has .npy files
    np.save(target_dir / "mouse_1.npy", np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
    np.save(target_dir / "mouse_2.npy", np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))

    with pytest.raises(FileNotFoundError, match="No files"):
        find_paired_files(str(input_dir), str(target_dir))

def test_find_paired_files_missing_npyfiles_raises_for_target(tmp_path):

    """
    GIVEN: target directory with no file that matching the pattern of *.npy
    WHEN: find_paired_files is called
    THEN: a FileNotFoundError is raised
    """

    # create directories
    input_dir =  tmp_path / "input"
    target_dir = tmp_path / "target"

    input_dir.mkdir()
    target_dir.mkdir()

    # save the .npy files in the directories
    # input directory has .npy files
    np.save(input_dir / "mouse_1.npy", np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
    np.save(input_dir / "mouse_2.npy", np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))

    # target directory has no .npy files, but has a .txt file
    (target_dir / "mouse.txt").write_text("hello world\n")

    with pytest.raises(FileNotFoundError, match="No files"):
        find_paired_files(str(input_dir), str(target_dir))

def test_find_paired_files_missing_files_raises_for_input(tmp_path):

    """
    GIVEN: target directory has .npy files but input directory is empty
    WHEN: find_paired_files is called
    THEN: a FileNotFoundError is raised
    """

    # create directories
    input_dir =  tmp_path / "input"
    target_dir = tmp_path / "target"

    input_dir.mkdir()
    target_dir.mkdir()

    # save the .npy files in the directories
    # input directory is left empty for this test.
    np.save(target_dir / "mouse_1.npy", np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
    np.save(target_dir / "mouse_2.npy", np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))

    with pytest.raises(FileNotFoundError, match="No files"):
        find_paired_files(str(input_dir), str(target_dir))

def test_find_paired_files_missing_files_raises_for_target(tmp_path):

    """
    GIVEN: input directory has .npy files but target directory is empty
    WHEN: find_paired_files is called
    THEN: a FileNotFoundError is raised
    """

    # create directories
    input_dir =  tmp_path / "input"
    target_dir = tmp_path / "target"

    input_dir.mkdir()
    target_dir.mkdir()

    # save the .npy files in the directories
    np.save(input_dir / "mouse_1.npy", np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
    np.save(input_dir / "mouse_2.npy", np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
    # target directory is left empty for this test.

    with pytest.raises(FileNotFoundError, match="No files"):
        find_paired_files(str(input_dir), str(target_dir))


def test_find_paired_files_count_mismatch_raises(tmp_path):

    """
    GIVEN: an input directory and a target directory with a different number of files
    WHEN: find_paired_files is called
    THEN: a ValueError is raised
    """

    # create directories
    input_dir =  tmp_path / "input"
    target_dir = tmp_path / "target"

    input_dir.mkdir()
    target_dir.mkdir()

    # Save .npy files in the directories
    for i in range(3):
        inp_file = input_dir / f"mouse_{i}.npy"
        np.save(inp_file, np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))

    for i in range(2):
        tg_file = target_dir / f"mouse_{i}.npy"
        np.save(tg_file, np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))

    with pytest.raises(ValueError, match="differ"):
        find_paired_files(str(input_dir), str(target_dir))


def test_find_paired_files_returns_sorted_pairs(tmp_path):

    """
    GIVEN: input and target directories with matching file names
    WHEN: find_paired_files is called
    THEN: it returns the paired files, sorted by name of .npy file
    """

    # make long path to directory containing files
    input_dir = tmp_path / "sub_folder1" / "sub_folder2" / "input"
    target_dir = tmp_path / "sub_folder1" / "sub_folder2" / "target"

    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Save .npy files in the directories
    np.save(input_dir / "mouse_01.npy", np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
    np.save(input_dir / "mouse_02.npy", np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
    np.save(target_dir / "mouse_01.npy", np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
    np.save(target_dir / "mouse_02.npy", np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))

    input_files, target_files = find_paired_files(str(input_dir), str(target_dir))

    # extract the names
    input_names = [os.path.basename(name) for name in input_files]
    target_names = [os.path.basename(name) for name in target_files]

    assert input_names == ["mouse_01.npy", "mouse_02.npy"], \
        "Input files should be sorted by name"
    assert target_names == ["mouse_01.npy", "mouse_02.npy"], \
        "Target files should be sorted by name"


def test_find_paired_files_name_mismatch_raises(tmp_path):

    """
    GIVEN: an input directory and a target directory with differently named files
    WHEN: find_paired_files is called
    THEN: a ValueError is raised
    """

    # make long path to directory containing files
    input_dir = tmp_path / "sub_folder1" / "sub_folder2" / "input"
    target_dir = tmp_path / "sub_folder1" / "sub_folder2" / "target"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Save .npy files in input and target directories while the names differ
    np.save(input_dir / "mouse_01.npy", np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
    np.save(input_dir / "mouse_02.npy", np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
    np.save(target_dir / "mouse_01.npy", np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
    np.save(target_dir / "mouse_03.npy", np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))

    with pytest.raises(ValueError, match="match by name"):
        find_paired_files(str(input_dir), str(target_dir))


def test_split_files_rounding_matches_hand_computed_sizes_secondhalfdecimal():
    """
    GIVEN: a list of 7 file pairs and a validation split of 0.25
    WHEN: split_files is called
    THEN: 2 files go to validation and 5 to training

    calculation by hand: 
    
        # 7 files x 0.25 = 1.75 (0.75 is in the second half between 1 and 2)
        the validation size is rounded up, ceil(0.25 * 7) = ceil(1.75) = 2, and the training
        size is the rest = 5.
    """
    number_of_files = 7
    # splitting 7 files with 0.25 yields decimal
    validation_split= 0.25 

    input_files = [f"mouse_{i}.npy" for i in range(number_of_files)]
    target_files = [f"mouse_{i}.npy" for i in range(number_of_files)]

    # split input and target to training set and validation set with
    # split fration of 0.25 (validation is 0.25)
    train_in, train_tgt, val_in, val_tgt = split_files(input_files, target_files, validation_split=validation_split, seed=42)

    assert (len(train_in), len(val_in)) == (5, 2), f"Expected 5 training and 2 validation files, got {len(train_in)} and {len(val_in)}"
    assert (len(train_tgt), len(val_tgt)) == (5, 2), f"Expected 5 training and 2 validation files, got {len(train_tgt)} and {len(val_tgt)}"
    
def test_split_files_rounding_matches_hand_computed_sizes_firsthalfdecimal():
    """
    GIVEN: a list of 7 file pairs and a validation split of 0.2
    WHEN: split_files is called
    THEN: 2 files go to validation and 5 to training

    calculation by hand: 

    # 7 files x 0.2 = 1.4 (0.4 is in the first half between 1 and 2)
    the validation size is rounded up, ceil(0.2 * 7) = ceil(1.4) = 2, and the training
    size is the rest = 5.
    """
    number_of_files = 7
    # splitting 7 files with 0.2 yields decimal
    validation_split = 0.2 

    input_files = [f"mouse_{i}.npy" for i in range(number_of_files)]
    target_files = [f"mouse_{i}.npy" for i in range(number_of_files)]

    # split input and target to training set and validation set with
    # split fration of 0.2 (validation is 0.2)
    train_in, train_tgt, val_in, val_tgt = split_files(input_files, target_files, validation_split=validation_split, seed=42)

    assert (len(train_in), len(val_in)) == (5, 2), f"Expected 5 training and 2 validation files, got {len(train_in)} and {len(val_in)}"
    assert (len(train_tgt), len(val_tgt)) == (5, 2), f"Expected 5 training and 2 validation files, got {len(train_tgt)} and {len(val_tgt)}"
    
def test_split_files_coverage():

    """
    GIVEN: a list of paired files and a validation split fraction
    WHEN: split_files is called
    THEN: the train and validation sets should cover the whole datasets after splitting
    """

    # create the list of inputs and targets paths
    input_files = [f"mouse_{i}.npy" for i in range(21)]
    target_files = [f"mouse_{i}.npy" for i in range(21)]

    # split input and target to training set and validation set with split fration of 0.25 (validation is 0.25, training is 0.75)
    train_inp, train_tg, val_inp, val_tg = split_files(input_files, target_files, validation_split=0.25, seed=42)

    assert set(train_inp) | set(val_inp) == set(input_files), "Train and validation sets of inputs should cover all inputs dataset"
    assert set(train_tg) | set(val_tg) == set(target_files), "Train and validation sets of targets should cover all targets dataset"


def test_split_files_overlap():

    """
    GIVEN: a list of paired files and a validation split fraction
    WHEN: split_files is called
    THEN: the train and validation sets shouldn't have overlap
    """

    # create the list of inputs and targets paths
    input_files = [f"mouse_{i}.npy" for i in range(21)]
    target_files = [f"mouse_{i}.npy" for i in range(21)]

    # split input and target to training set and validation set with split fration of 0.25 (validation is 0.25, training is 0.75)
    train_inp, train_tg, val_inp, val_tg = split_files(input_files, target_files, validation_split=0.25, seed=42)

    assert set(train_inp).isdisjoint(set(val_inp)), "Train and validation sets of inputs should not overlap"
    assert set(train_tg).isdisjoint(set(val_tg)), "Train and validation sets of targets should not overlap"


def test_split_files_deterministic_with_seed():

    """
    GIVEN: the same seed used twice
    WHEN: split_files is called
    THEN: the same train/validation split is produced both times
    """

    # create the list of inputs and targets paths
    input_files = [f"mouse_{i}.npy" for i in range(10)]
    target_files = [f"mouse_{i}.npy" for i in range(10)]

    split_files_1 = split_files(input_files, target_files, validation_split=0.25, seed=42)
    split_files_2 = split_files(input_files, target_files, validation_split=0.25, seed=42)

    assert split_files_1 == split_files_2, "Same seed should produce the same train/validation split"


def test_collate_batch_returns_tensor(tmp_path):
    """
    GIVEN: a list of (input_path, target_path) pairs
    WHEN: collate_batch is called
    THEN: the stacked data (batch) of inputs and targets are tensors
    """
    
    # create directories and files
    input_dir =  tmp_path / "input"
    target_dir = tmp_path / "target"
    input_dir.mkdir()
    target_dir.mkdir()
    
    path_pairs = []
    # consider batch size is 8
    batch_size = 8
    for i in range(batch_size):
        inp_file = input_dir / f"mouse_{i}.npy"
        np.save(inp_file, np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
        tg_file = target_dir / f"mouse_{i}.npy"
        np.save(tg_file, np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
        path_pairs.append((inp_file, tg_file))
        
    inputs, targets = collate_batch(path_pairs)
    
    assert isinstance(inputs, torch.Tensor), "collate_batch should return a torch.Tensor"
    assert isinstance(targets, torch.Tensor), "collate_batch should return a torch.Tensor"


def test_collate_batch_stacks_pairs(tmp_path):
    """
    GIVEN: a list of (input_path, target_path) pairs
    WHEN: collate_batch is called
    THEN: the stacked data (batch) of inputs and targets have
          shape of (Batch size, number of Channels=2, H=300, W=300)
    """

    # create directories and files
    input_dir =  tmp_path / "input"
    target_dir = tmp_path / "target"
    input_dir.mkdir()
    target_dir.mkdir()

    path_pairs = []
    # consider batch size is 8
    batch_size = 8
    for i in range(batch_size):
        inp_file = input_dir / f"mouse_{i}.npy"
        np.save(inp_file, np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
        tg_file = target_dir / f"mouse_{i}.npy"
        np.save(tg_file, np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
        path_pairs.append((inp_file, tg_file))

    inputs, targets = collate_batch(path_pairs)

    assert inputs.shape == (batch_size, 2, 300, 300), f"Unexpected inputs shape: {tuple(inputs.shape)}"
    assert targets.shape == (batch_size, 2, 300, 300), f"Unexpected targets shape: {tuple(targets.shape)}"

    
def test_collate_batch_casts_to_float32(tmp_path):
    """
    GIVEN: .npy files in float 16 and float64
    WHEN: load_projection is called
    THEN: the returned tensor is float32, as the model expects
    """
    
    # create directories and files
    input_dir =  tmp_path / "input"
    target_dir = tmp_path / "target"
    input_dir.mkdir()
    target_dir.mkdir()
    
    path_pairs = []
    path_inp = input_dir / "mouse_1.npy"
    np.save(path_inp, np.full(IMAGE_SHAPE, 0.5, dtype=np.float16))
    path_tg = target_dir / "mouse_1.npy"
    np.save(path_tg, np.full(IMAGE_SHAPE, 0.5, dtype=np.float64))
    path_pairs.append((path_inp, path_tg))

    inputs, targets = collate_batch(path_pairs)

    assert inputs.dtype == torch.float32, f"Expected float32, got {inputs.dtype}"
    assert targets.dtype == torch.float32, f"Expected float32, got {targets.dtype}"

# ----------------------------------------------------
# tests for collate_augmented_batch function
# ----------------------------------------------------

def test_collate_augmented_batch_stacks_pairs(tmp_path):
    """
    GIVEN: a list of (input_path, target_path) pairs
    WHEN: collate_augmented_batch is called
    THEN: the stacked data (batch) of inputs and targets have
          shape of (Batch size, number of Channels=2, H=300, W=300)
    """

    # create directories and files
    input_dir =  tmp_path / "input"
    target_dir = tmp_path / "target"
    input_dir.mkdir()
    target_dir.mkdir()

    path_pairs = []
    # consider batch size is 8
    batch_size = 8
    for i in range(batch_size):
        inp_file = input_dir / f"mouse_{i}.npy"
        np.save(inp_file, np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
        tg_file = target_dir / f"mouse_{i}.npy"
        np.save(tg_file, np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
        path_pairs.append((inp_file, tg_file))

    inputs, targets = collate_augmented_batch(path_pairs, use_augmentation=False)

    assert inputs.shape == (batch_size, 2, 300, 300), f"Unexpected inputs shape: {tuple(inputs.shape)}"
    assert targets.shape == (batch_size, 2, 300, 300), f"Unexpected targets shape: {tuple(targets.shape)}"


@pytest.mark.parametrize("path_pairs", ALL_PATH_PAIRS)
def test_collate_augmented_batch_acts_like_collate_batch_when_data_augmentation_is_disabled(path_pairs):
    """
    GIVEN: batch of mouse projections while data augmentation is disabled but
           rotation and flipping has probability of 1
    WHEN: collate_augmented_batch is called
    THEN: the batch returned by collate_augmented_batch equals
          the one returned by collate_batch (it means no transformation).
    """

    path_pairs = [path_pairs]

    expected_inputs, expected_targets = collate_batch(path_pairs)

    inputs, targets = collate_augmented_batch(
        path_pairs,
        use_augmentation=False,
        rotation_probability=1.0,
        flip_probability=1.0
    )

    assert torch.equal(inputs, expected_inputs), \
        "The inputs changed although data augmentation is disabled"

    assert torch.equal(targets, expected_targets), \
        "The targets changed although data augmentation is disabled"

@pytest.mark.parametrize("path_pairs", ALL_PATH_PAIRS)
def test_collate_augmented_batch_acts_like_collate_batch_when_zero_probabilities(path_pairs):
    """
    GIVEN: batch of mouse projections while rotation and flipping probability are 0
           but data augmentation is enabled.
    WHEN: collate_augmented_batch is called.
    THEN: the batch returned by collate_augmented_batch equals
          the one returned by collate_batch (it means no transformation).
    """
    
    path_pair = [path_pairs]

    expected_inputs, expected_targets = collate_batch(path_pair)

    inputs, targets = collate_augmented_batch(
        path_pair,
        use_augmentation=True,
        rotation_probability=0.0,
        flip_probability=0.0
    )

    assert torch.equal(inputs, expected_inputs), "The inputs changed although the rotation and flipping probability is 0"
    assert torch.equal(targets, expected_targets), "The targets changed although the rotation and flipping probability is 0"

@pytest.mark.parametrize("path_pair", ALL_PATH_PAIRS[:1]) 
# only test on the first projection pair to avoid long test time since it is 100 copies of the same projection
def test_collate_augmented_batch_rotates_expected_number_of_projections(path_pair):
    """
    GIVEN: a batch of 100 copies of one mouse projection while only rotation is enabled
           with a probability of 0.25 for augmentation
    WHEN: collate_augmented_batch is called with a seeded generator
    THEN: about 25% of the projections are rotated
    """

    path_pairs = [path_pair]*100  # create a batch of 100 copies of the same projection
    # does not matter which projection is used, we just want to check the share of rotated projections
    original_inputs, _ = collate_batch(path_pairs)

    # Seed the random generator to make the test reproducible
    random.seed(42)

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



@pytest.mark.parametrize("path_pair", ALL_PATH_PAIRS[:1]) # only test on the first pair to avoid long test time
def test_collate_augmented_batch_flips_about_the_expected_share_of_projections(path_pair):
    """
    GIVEN: a batch of 100 copies of one mouse projection while only flipping is enabled
           with a probability of 0.25 for augmentation
    WHEN: collate_augmented_batch is called with a seeded generator
    THEN: about 25% of the projections are flipped
    """

    path_pair = [path_pair]
    original_inputs, _ = collate_batch(path_pair)

    random.seed(42)

    inputs, _ = collate_augmented_batch(
        path_pair*100,
        use_augmentation=True,
        rotation_probability=0.0,
        flip_probability=0.25
    )

    n_flipped = 0
    for projection in inputs:
                if not torch.equal(projection, original_inputs[0]):
                    n_flipped += 1

    assert abs(n_flipped / 100 - 0.25) < 0.15, f"Expected about 25% flipped projections, got {n_flipped}%"


@pytest.mark.parametrize("path_pair", ALL_PATH_PAIRS[:1]) 
# only test on the first pair to avoid long test time
def test_collate_augmented_batch_uses_every_rotation_angle(path_pair):

    """
    GIVEN: a batch of many copies of the same mouse projection while
           only the rotation enabled with a probability of 1.
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
    
    random.seed(42)

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

@pytest.mark.parametrize("path_pair", ALL_PATH_PAIRS[:1]) # only test on the first pair to avoid long test time
def test_collate_augmented_batch_uses_every_flip_type(path_pair):
    """
    GIVEN: only the flipping enabled with a probability of 1, and a batch of 100 copies of the
           same real projection
    WHEN: collate_augmented_batch is called with a seeded generator
    THEN: the horizontal, the vertical and the double flip all occur in the batch
    """

    path_pairs = [path_pair]
    original_inputs, _ = collate_batch(path_pairs)

    # The 3 possible cases of flipping a (C, H, W) tensor, in the order of FLIP_TYPES.
    flip_case1 = torch.flip(original_inputs[0], dims=[1])
    flip_case2 = torch.flip(original_inputs[0], dims=[2])
    flip_case3 = torch.flip(original_inputs[0], dims=[1, 2])
    all_flip_cases = [flip_case1, flip_case2, flip_case3]

    random.seed(42)

    inputs, _ = collate_augmented_batch(
        path_pairs*100,
        use_augmentation=True,
        rotation_probability=0.0,
        flip_probability=1.0
    )

    flips_found = []
    for projection in inputs:
        for flip_type in FLIP_TYPES:
            if torch.equal(projection, all_flip_cases[flip_type]):
                if flip_type not in flips_found:
                    flips_found.append(flip_type)

    assert 0 in flips_found, "Flip type 0 (horizontal flip) was never found in the batch"
    assert 1 in flips_found, "Flip type 1 (vertical flip) was never found in the batch"
    assert 2 in flips_found, "Flip type 2 (double flip) was never found in the batch"


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

    random.seed(42)

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


    # With 300 projections the standard deviation of each proportion is:
    # std = sqrt( p * (1-p) / n ) = sqrt((1/3)*(2/3)/300) = sqrt(0.000741) = 0.027,
    # in a normal approximation, ±2.6*std covers about 99% of outcomes => 2.6*0.027 = 0.07, which is 7 percent
    # so a tolerance of 0.07 lets the test passes for any fixed random seed
    tolerance = 0.07  # allow +/- 7 percentage points of random noise

    assert abs(proportion[1] - expected_proportion) < tolerance, (
                f"Rotation k=1 occurred {proportion[1]:.2%} of the time, "
                f"expected close to {expected_proportion:.2%}")
   
    assert abs(proportion[2] - expected_proportion) < tolerance, (
                f"Rotation k=2 occurred {proportion[2]:.2%} of the time, "
                f"expected close to {expected_proportion:.2%}")
    
    assert abs(proportion[3] - expected_proportion) < tolerance, (
                f"Rotation k=3 occurred {proportion[3]:.2%} of the time, "
                f"expected close to {expected_proportion:.2%}")

@pytest.mark.parametrize("path_pair", ALL_PATH_PAIRS[:1]) 
# only test on the first pair to avoid long test time
def test_collate_augmented_batch_flip_distribution_is_uniform(path_pair):
    """
    GIVEN: a batch of many copies of the same mouse projection while
           the flipping enabled with a probability of 1.
    WHEN: collate_augmented_batch is called with a seeded generator
    THEN: each of the 3 possible flips (horizontal, vertical, both) occurs roughly 1/3 of the time,
          confirming the flip type is chosen uniformly at random for each projection
    """

    path_pair = [path_pair]
    original_inputs, _ = collate_batch(path_pair)

    # The 3 possible cases of flipping a (C, H, W) tensor, in the order of FLIP_TYPES.
    flip_case0 = torch.flip(original_inputs[0], dims=[2])     # horizontal flip: along the width
    flip_case1 = torch.flip(original_inputs[0], dims=[1])     # vertical flip: along the height
    flip_case2 = torch.flip(original_inputs[0], dims=[1, 2])  # both flips
    all_flip_cases = [flip_case0, flip_case1, flip_case2]

    random.seed(42)

    batch_size = 300  # large enough for the counts to settle near the true probability
    inputs, _ = collate_augmented_batch(
            path_pair*batch_size,
            use_augmentation=True,
            rotation_probability=0.0,
            flip_probability=1.0
        ) # rotation is disabled to avoid interference with flip counts

    # Count how many times each flip type was found in the batch.
    flips_found = {0: 0, 1: 0, 2: 0}
    for projection in inputs:
        for flip_type in FLIP_TYPES:
            if torch.equal(projection, all_flip_cases[flip_type]):
                flips_found[flip_type] += 1

    # Convert counts into proportions out of the whole batch.
    proportion = {flip_type: flips_found[flip_type] / batch_size for flip_type in FLIP_TYPES}

    expected_proportion = 1 / len(FLIP_TYPES)  # should be 1/3 for all flip types

    # With 300 projections the standard deviation of each proportion is:
    # std = sqrt( p * (1-p) / n ) = sqrt((1/3)*(2/3)/300) = sqrt(0.000741) = 0.027,
    # in a normal approximation, ±2.6*std covers about 99% of outcomes => 2.6*0.027 = 0.07, which is 7 percent
    # so a tolerance of 0.07 lets the test passes for any fixed random seed
    tolerance = 0.07  # allow +/- 7 percentage points of random noise

    assert abs(proportion[0] - expected_proportion) < tolerance, (
                f"Horizontal flip occurred {proportion[0]:.2%} of the time, "
                f"expected close to {expected_proportion:.2%}")

    assert abs(proportion[1] - expected_proportion) < tolerance, (
                f"Vertical flip occurred {proportion[1]:.2%} of the time, "
                f"expected close to {expected_proportion:.2%}")

    assert abs(proportion[2] - expected_proportion) < tolerance, (
                f"Both flips occurred {proportion[2]:.2%} of the time, "
                f"expected close to {expected_proportion:.2%}")
