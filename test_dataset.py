
import os
import numpy as np
import pytest
from dataset import (check_files, find_paired_files, split_files)

IMAGE_SHAPE = (300, 300, 2)  # (H, W, C) for each single projection

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


def test_check_files_length_mismatch_raises(tmp_path):

    """
    GIVEN: input is 3 files but target is 2 files
    WHEN: check_files is called
    THEN: a ValueError is raised
    """

    # create directories and files

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


def test_check_files_wrong_shape_raises(tmp_path):

    """
    GIVEN: a .npy file as input whose shape is not the expected shape i.e. (300, 300, 2)
    WHEN: check_files is called
    THEN: a ValueError naming the expected shape is raised
    """

    # create directories and files

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

def test_find_paired_files_missing_files_raises(tmp_path):

    """
    GIVEN: target directory with no file that matching the pattern of *.npy
    WHEN: find_paired_files is called
    THEN: a FileNotFoundError is raised
    """

    # create directories and files
    input_dir =  tmp_path / "input"
    target_dir = tmp_path / "target"

    input_dir.mkdir()
    target_dir.mkdir()

    np.save(input_dir / "mouse_1.npy", np.eye(3))
    np.save(input_dir / "mouse_2.npy", np.eye(3))
    # target directory is left empty

    with pytest.raises(FileNotFoundError, match="No files"):
        find_paired_files(str(input_dir), str(target_dir))


def test_find_paired_files_count_mismatch_raises(tmp_path):

    """
    GIVEN: an input directory and a target directory with a different number of files
    WHEN: find_paired_files is called
    THEN: a ValueError is raised
    """

    # create directories and files

    input_dir =  tmp_path / "input"
    target_dir = tmp_path / "target"

    input_dir.mkdir()
    target_dir.mkdir()

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

    # Save .npy files in input and target
    np.save(input_dir / "mouse_01.npy", np.eye(3))
    np.save(input_dir / "mouse_02.npy", np.eye(3))

    np.save(target_dir / "mouse_01.npy", np.eye(3))
    np.save(target_dir / "mouse_02.npy", np.eye(3))

    input_files, target_files = find_paired_files(str(input_dir), str(target_dir))

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

    # Save .npy files in input and target folders while the names differ
    np.save(input_dir / "mouse_01.npy", np.eye(3))
    np.save(input_dir / "mouse_02.npy", np.eye(3))

    np.save(target_dir / "mouse_01.npy", np.eye(3))
    np.save(target_dir / "mouse_03.npy", np.eye(3))

    with pytest.raises(ValueError, match="match by name"):
        find_paired_files(str(input_dir), str(target_dir))


def test_split_files_sizes_and_coverage():

    """
    GIVEN: a list of paired files and a validation split fraction
    WHEN: split_files is called
    THEN: the train/validation sets partition all files with the expected sizes
    """

    # create list of 21 files (splitting with 0.25 yields decimal) for input and target
    input_files = [f"mouse_{i}.npy" for i in range(21)]
    target_files = [f"mouse_{i}.npy" for i in range(21)]

    # split input and target to training set and validation set with fration of 0.25
    train_inp, train_tg, val_inp, val_tg = split_files(input_files, target_files, validation_split=0.25, seed=42)

    # for deciama values it ceils (ceil(5.25) = 6)
    assert len(train_inp) == len(train_tg) == 15, f"Expected 15 training files, got {len(train_inp)}"
    assert len(val_inp) == len(val_tg) == 6, f"Expected 6 validation files, got {len(val_inp)}"
    assert set(train_inp) | set(val_inp) == set(input_files), "Train/validation split should cover all files"
    assert set(train_inp).isdisjoint(set(val_inp)), "Train and validation sets should not overlap"


def test_split_files_deterministic_with_seed():

    """
    GIVEN: the same seed used twice
    WHEN: split_files is called
    THEN: the same train/validation split is produced both times
    """

    input_files = [f"mouse_{i}.npy" for i in range(10)]
    target_files = [f"mouse_{i}.npy" for i in range(10)]

    split_files_1 = split_files(input_files, target_files, validation_split=0.2, seed=42)
    split_files_2 = split_files(input_files, target_files, validation_split=0.2, seed=42)

    assert split_files_1 == split_files_2, "Same seed should produce the same train/validation split"

