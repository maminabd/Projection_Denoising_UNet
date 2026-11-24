
import torch
import pytest
import torch.nn as nn
import torch.optim as optim
from libs.utils import (absolute_difference, mean_absolute_difference, projs_comparison,
                        load_checkpoint, save_checkpoint)


def test_mean_absolute_difference_returns_handy_calculated_values():
    """
    GIVEN: denoised tensor and its ground truth tensor in shape of (2,2,2) which is (channels, height, width)
    WHEN: mean_absolute_difference is called
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

    mean_differences = mean_absolute_difference(denoised, ground_truth)

    assert mean_differences == pytest.approx([0.1875, 1.0]), f"Expected [0.1875, 1.0], got {mean_differences}"

def test_mean_absolute_difference_returns_list():
    """
    GIVEN: denoised tensor and its ground truth tensor in shape of (2, 300, 300) which is (channels, height, width)
    WHEN: mean_absolute_difference is called
    THEN: it returns a python list
    """
    denoised = torch.full((2, 300, 300), 0.55)
    ground_truth = torch.full((2, 300, 300), 0.5)

    mean_differences = mean_absolute_difference(denoised, ground_truth)

    assert isinstance(mean_differences, list), f"Expected a python list, got {type(mean_differences)}"


def test_mean_absolute_difference_returns_2elements_list():
    """
    GIVEN: denoised tensor and its ground truth tensor in shape of (2, 300, 300) which is (channels, height, width)
    WHEN: mean_absolute_difference is called
    THEN: the returned list should have 2 elements, 
          each element is the mean absolute difference of the corresponding channel of LE and HE
    """
    denoised = torch.full((2, 300, 300), 0.55)
    ground_truth = torch.full((2, 300, 300), 0.5)

    mean_differences = mean_absolute_difference(denoised, ground_truth)

    assert len(mean_differences) == 2, (f"Expected exactly 2 elements (one per energy channel), got {len(mean_differences)}")


def test_absolute_difference_is_symmetric():
    """
    GIVEN: two tensors of shape (2, 2, 2)
    WHEN: absolute_difference is called with the tensors in both orders
    THEN: the two difference maps are equal, as the sign of the difference does not matter
    """
    
    denoised = torch.full((2, 300, 300), 0.55)
    ground_truth = torch.full((2, 300, 300), 0.5)

    assert torch.equal(absolute_difference(denoised, ground_truth), absolute_difference(ground_truth, denoised)), \
        "The absolute difference should not depend on the order of the tensors"


def test_absolute_difference_keeps_the_shape():
    """
    GIVEN: denoised tensor and its ground truth tensor in shape of (2, 300, 150)
    WHEN: absolute_difference is called
    THEN: the difference map has the same shape (2, 300, 150) as the input tensors
    """
    difference = absolute_difference(torch.rand(2, 300, 150), torch.rand(2, 300, 150))

    assert difference.shape == (2, 300, 150), f"Expected shape (2, 300, 150), got {tuple(difference.shape)}"

def test_save_and_load_checkpoint(tmp_path):
    
    """
    GIVEN: a model, optimizer and scheduler whose state is saved to a checkpoint
    WHEN: load_checkpoint is called on a fresh model/optimizer/scheduler
    THEN: the restored checkpointed_epoch and checkpointed_val_loss match what was saved
    """

    model = nn.Linear(4, 2) # make a simple model (a linear) for testing
    optimizer = optim.AdamW(model.parameters(), lr=1e-3) # make a simple optimizer for testing
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1) # make a simple scheduler for testing
    checkpoint_path = str(tmp_path / "checkpoint.pth") # temporary path for saving the checkpoint

    # Save the checkpoint with some values for checkpointed_epoch and checkpointed_val_loss
    save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch=4, train_loss = 0.1, val_loss=0.1234)

    new_model = nn.Linear(4, 2) # create a new model with the same architecture
    new_optimizer = optim.AdamW(new_model.parameters(), lr=1e-3) # create a new optimizer for the new model
    new_scheduler = optim.lr_scheduler.StepLR(new_optimizer, step_size=1) # create a new scheduler for the new optimizer

    # load the saved checkpoint into the new model/optimizer/scheduler 
    checkpointed_epoch, checkpointed_val_loss = load_checkpoint(checkpoint_path, new_model, new_optimizer,
                                                  new_scheduler, device=torch.device('cpu'))

    assert checkpointed_epoch == 5, f"Expected to resume from epoch 5 (one after the checkpointed epoch), got {checkpointed_epoch}"

    assert checkpointed_val_loss == pytest.approx(0.1234), f"Expected val_loss 0.1234, got {checkpointed_val_loss}"
    
    for params_saved, params_loaded in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(params_saved, params_loaded), "Restored model parameters don't match the saved ones"

def test_load_checkpoint_restores_optimizer_and_scheduler_state(tmp_path):
    """
    GIVEN: an optimizer and a scheduler that have been stepped three times (lr = 0.1 * 0.5^3 = 0.0125)
    WHEN: their checkpoint is loaded into a fresh optimizer and scheduler
    THEN: the fresh learning rate is 0.0125 too, so a resumed run continues the schedule

    By hand: StepLR(step_size=1, gamma=0.5) halves the learning rate at every step.
    """
    model = nn.Linear(3, 2) # make a simple model (a linear) for testing
    
    optimizer = optim.SGD(model.parameters(), lr=0.1) # make a simple optimizer for testing

    # make a simple scheduler for testing
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5) # at each step halves the learning rate

    # Step the optimizer and scheduler three times to change the learning rate
    for _ in range(3):
        optimizer.step() # update the model parameters based on the current gradients.
        scheduler.step() # update the learning rate according to the schedule (halving it each time)
    path = str(tmp_path / "scheduled.pth") # temporary path for saving the checkpoint

    # Save the checkpoint with the current state of the model, optimizer, and scheduler
    save_checkpoint(path, model, optimizer, scheduler, epoch=2)

    # Create new instances of the model, optimizer, and scheduler to load the checkpoint into
    new_model = nn.Linear(3, 2)  # make a new model with the same architecture 
    new_optimizer = optim.SGD(new_model.parameters(), lr=0.1) # make a new optimizer for the new model with the same initial learning rate
    new_scheduler = optim.lr_scheduler.StepLR(new_optimizer, step_size=1, gamma=0.5) # make a new scheduler for the new optimizer with the same schedule
    
    # Load the checkpoint into the new model, optimizer, and scheduler
    load_checkpoint(path, new_model, new_optimizer, new_scheduler, torch.device('cpu'))

    assert new_optimizer.param_groups[0]['lr'] == pytest.approx(0.0125), \
        f"Expected the resumed learning rate 0.0125, got {new_optimizer.param_groups[0]['lr']}"