import numpy as np
import pytest
from dataset import check_files

IMAGE_SHAPE = (300, 300, 2)  # (H, W, C) for each single projection

def test_check_files_accepts_valid_files(tmp_path):
    """
    GIVEN: paired files that all have the expected shape and both input and target have the same length
    WHEN: check_files is called
    THEN: no error is raised
    """
    input_files = []
    for i in range(2):
        path = tmp_path / f"in_{i}.npy"
        np.save(path, np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
        input_files.append(str(path))

    target_files = []
    for i in range(2):
        path = tmp_path / f"tg_{i}.npy"
        np.save(path, np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
        target_files.append(str(path))

    check_files(input_files, target_files)


def test_check_files_length_mismatch_raises(tmp_path):
    """
    GIVEN: input is 3 files but target is 2 files
    WHEN: check_files is called
    THEN: a ValueError is raised
    """
    input_files = []
    for i in range(3):
        path = tmp_path / f"in_{i}.npy"
        np.save(path, np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
        input_files.append(str(path))

    target_files = []
    for i in range(2):
        path = tmp_path / f"tg_{i}.npy"
        np.save(path, np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
        target_files.append(str(path))

    with pytest.raises(ValueError, match="differ"):
        check_files(input_files, target_files)


def test_check_files_wrong_shape_raises(tmp_path):
    """
    GIVEN: a file whose shape is not excepted shape i.e. (300, 300, 2)
    WHEN: check_files is called
    THEN: a ValueError naming the expected shape is raised
    """
    in_path = tmp_path / "in_0.npy"
    np.save(in_path, np.full((300, 300), 0.5, dtype=np.float32))
    input_files = [str(in_path)]

    tg_path = tmp_path / "tg_0.npy"
    np.save(tg_path, np.full(IMAGE_SHAPE, 0.5, dtype=np.float32))
    target_files = [str(tg_path)]

    with pytest.raises(ValueError, match="expected shape"):
        check_files(input_files, target_files)

