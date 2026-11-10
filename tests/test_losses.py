
import torch
from libs.losses import denoising_loss, gradient_loss

def test_gradient_loss_is_zero_for_identical_tensors():
    """
    GIVEN: two identical projection batches
    WHEN: gradient_loss is computed
    THEN: the loss is zero, since their gradients are identical
    """
    denoised = torch.full((4, 2, 300, 300), 0.5)
    target = torch.full((4, 2, 300, 300), 0.5)


    grad_loss = gradient_loss(denoised, target)

    assert torch.isclose(grad_loss, torch.tensor(0.0), atol=1e-7), \
        f"Expected ~0 for identical tensors, got {grad_loss.item()}"

def test_denoising_loss_is_zero_for_identical_tensors():
    """
    GIVEN: two identical projection batches
    WHEN: denoising_loss is called
    THEN: the totla loss is zero
    """

    denoised = torch.full((4, 2, 300, 300), 0.5)
    target = torch.full((4, 2, 300, 300), 0.5)

    total_loss = denoising_loss(denoised, target)

    assert torch.isclose(total_loss, torch.tensor(0.0), atol=1e-7), \
            f"Expected ~0 for identical tensors, got {total_loss.item()}"


def test_gradient_loss_ignores_a_constant_offset():
    """
    GIVEN: a prediction batch which is the target batch plus a constant
    WHEN: gradient_loss is computed
    THEN: the gradient loss is zero because a constant offset should not change the
          gradients (this is what makes this loss component an edge preserver term, not an intensity term)
    """

    target = torch.full((4, 2, 300, 300), 0.5)
    prediction = target + 0.3  # add a constant offset

    loss = gradient_loss(prediction, target)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7), \
        f"A constant offset should not change the gradients, got {loss.item()}"

def test_gradient_loss_grows_with_noise():
    """
    GIVEN: a target batch with a sharp vertical straight line, and 
           a prediction batch made of the same line plus noise

    WHEN: gradient_loss is computed between prediction and the target
    THEN: the loss for the noisy prediction is larger than the loss for the noiseless prediction
    """
    target = torch.zeros(4, 2, 300, 300)
    target[:, :, :, 7] = 1.0  # a sharp straight line in column 7

    torch.manual_seed(42)
    noise = torch.randn(4, 2, 300, 300)
    noisy_prediction = target + 0.1 * noise

    grad_loss_noiseless = gradient_loss(target, target)
    grad_loss_noisy = gradient_loss(noisy_prediction, target)

    assert grad_loss_noiseless < grad_loss_noisy, f"With noise the loss should be larger than noiseless, got {grad_loss_noisy} < {grad_loss_noiseless}"


def test_gradient_loss_treats_height_and_width_alike():
    """
    GIVEN: one projection with a step between its two rows and 
           one projection with a step between its two columns,
           each compared with a zero target.

    WHEN: gradient_loss is computed
    THEN: both losses are equal
    """
    row_step_proj = torch.tensor([[0.0, 0.0], [1.0, 1.0]]).view(1, 1, 2, 2)
    column_step_proj = torch.tensor([[0.0, 1.0], [0.0, 1.0]]).view(1, 1, 2, 2)
    target = torch.zeros(1, 1, 2, 2)

    grad_loss_row = gradient_loss(row_step_proj, target)
    grad_loss_column = gradient_loss(column_step_proj, target)

    assert torch.isclose(grad_loss_row, grad_loss_column, atol=1e-7), "Gradient losses should be equal for row and column step projections."