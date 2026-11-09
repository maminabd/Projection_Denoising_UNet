
import numpy as np
import torch
import pytest
from libs.utils import projs_comparison

def test_save_comparison_returns_handy_calculated_values(tmp_path):
    """
    GIVEN: denoised tensor and its ground truth tensor in shape of (2,2,2) which is (channels, height, width)
    WHEN: save_comparison is called
    THEN: the returned mean absolute differences values should be [0.1875, 1.0] (calculated by hand)

    By hand:
      channel 0 (LE): denoised [[0, 1], [0.5, 0]], ground truth [[0.25, 1], [0, 0]]
                 |difference| = [[0.25, 0], [0.5, 0]], mean = 0.75/4 = 0.1875
      channel 1 (HE): denoised all are 0, ground truth all are 1, |difference| = all values are 1 , mean = 1.0
    """

    denoised = torch.tensor([[[0.0, 1.0], [0.5, 0.0]], 
                             [[0.0, 0.0], [0.0, 0.0]]])
    
    ground_truth = torch.tensor([[[0.25, 1.0], [0.0, 0.0]], 
                                 [[1.0,  1.0],  [1.0, 1.0]]])

    mean_differences = projs_comparison(denoised, ground_truth, str(tmp_path / "comparison.png"), "title")

    assert mean_differences == pytest.approx([0.1875, 1.0]), f"Expected [0.1875, 1.0], got {mean_differences}"

def test_save_comparison_returns_list(tmp_path):
    """
    GIVEN: denoised tensor and its ground truth tensor in shape of (2,2,2) which is (channels, height, width)
    WHEN: save_comparison is called
    THEN: it returns a python list
    """

    denoised = torch.full((2, 300, 300), 0.55)
    ground_truth = torch.full((2, 300, 300), 0.5)

    mean_differences = projs_comparison(denoised, ground_truth, str(tmp_path / "comparison.png"), "example")

    assert not isinstance(mean_differences, torch.Tensor), "Expected a python list, not a torch tensor"


def test_save_comparison_returns_2elements_list(tmp_path):
    """
    GIVEN: denoised tensor and its ground truth tensor in shape of (2,2,2) which is (channels, height, width)
    WHEN: save_comparison is called
    THEN: the returned list should have 2 elements, 
          each element is the mean absolute difference of the corresponding channel of LE and HE
    """

    denoised = torch.full((2, 300, 300), 0.55)
    ground_truth = torch.full((2, 300, 300), 0.5)

    mean_differences = projs_comparison(denoised, ground_truth, str(tmp_path / "comparison.png"), "example")

    assert len(mean_differences) == 2, (f"Expected exactly 2 elements (one per energy channel), got {len(mean_differences)}")