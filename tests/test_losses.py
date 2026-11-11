
import torch
from libs.losses import denoising_loss, gradient_loss

def test_gradient_loss_is_zero_for_identical_tensors():
    """
    GIVEN: two identical projection batches
    WHEN: gradient_loss is computed
    THEN: the loss is zero, since their gradients are identical
    """
    prediction = torch.full((4, 2, 300, 300), 0.5)
    target = torch.full((4, 2, 300, 300), 0.5)


    grad_loss = gradient_loss(prediction, target)

    assert torch.isclose(grad_loss, torch.tensor(0.0), atol=1e-7), \
        f"Expected ~0 for identical tensors, got {grad_loss.item()}"

def test_gradient_loss_swap_prediction_target():
    """
    GIVEN: two projection batches
    WHEN: gradient_loss is computed
    THEN: the gradient loss is the same regardless of order of entries 
          (i.e. gradient_loss(prediction, target) or gradient_loss(target, prediction))
          because it is based on the mean squared error of the gradients.
    """
    prediction = torch.full((4, 2, 300, 300), 0.51)
    target = torch.full((4, 2, 300, 300), 0.5)

    grad_loss_1 = gradient_loss(prediction, target)
    # swaped one
    grad_loss_2 = gradient_loss(target, prediction)

    assert torch.isclose(grad_loss_1, grad_loss_2, atol=1e-7), \
        f"Expected same value for swapped entries, got different values of {grad_loss_1.item()} and {grad_loss_2.item()}"

def test_denoising_loss_is_zero_for_identical_tensors():
    """
    GIVEN: two identical projection batches
    WHEN: denoising_loss is called
    THEN: the totla loss is zero
    """

    prediction = torch.full((4, 2, 300, 300), 0.5)
    target = torch.full((4, 2, 300, 300), 0.5)

    total_loss = denoising_loss(prediction, target)

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

    assert grad_loss_noiseless < grad_loss_noisy, \
        f"With noise the loss should be larger than noiseless, got {grad_loss_noisy} < {grad_loss_noiseless}"


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

    assert torch.isclose(grad_loss_row, grad_loss_column, atol=1e-7), \
        "Gradient losses should be equal for row and column step projections."

def test_gradient_loss_matches_handy_calculated_value():

    """
    GIVEN: a 4x10 target with a one-column line (column 7 = 1), and an prediction with a two-column
           line (column 7 = 1, column 8 = 0.25); the other pixels are 0
    WHEN: gradient_loss is computed
    THEN: it equals 0.125/18 = 0.00694444, the value calculated by hand below

    -height gradients
         All rows are identical, so the height gradients (differences between neighbouring rows)
         are 0 for both images and their MSE is 0.

    -width gradient
         The width gradient is the difference between neighbouring columns.
         dw[j] = row[j + 1] - row[j], giving 9 values per row for 10 columns:

         column (j): ... 5  6  7    8  9
         pred row:   ... 0  1  0.25 0  0 -> dw[j=5] = 0, dw[j=6] = 1, dw[j=7] = -0.75, dw[j=8] = -0.25
         target row: ... 0  1  0    0  0 -> dw[j=5] = 0, dw[j=6] = 1, dw[j=7] = -1,    dw[j=8] = 0

         All other dw[j] are 0 in both taget and prediction

      3. Difference of the gradients (prediction - target): 
                
            Diff[j] =  prediction_dw[j] - target_dw[j]

            Diff[j=0]: 0,
            Diff[j=1]: 0,
            Diff[j=2]: 0,
            Diff[j=3]: 0,
            Diff[j=4]: 0,
            Diff[j=5]: 0,
            Diff[j=5]: 0,
            Diff[j=6]: 1-1 = 0,
            Diff[j=7]: -0.75 + 1 = 0.25,
            Diff[j=8]: -0.25 - 0 = -0.25,
            Diff[j=9]: 0

      4. MSE of the width gradients = mean of the squares over the 9 values of a row
         (every row is the same) = (0.25^2 + (-0.25)^2) / 9 = 0.125 / 9

      5. gradient_loss = (MSE_height + MSE_width) / 2 = 0.125 / 18
    """

    target = torch.zeros(1, 1, 4, 10)
    target[:, :, :, 7] = 1.0

    pred = torch.zeros(1, 1, 4, 10)
    pred[:, :, :, 7] = 1.0
    pred[:, :, :, 8] = 0.25

    expected_loss = 0.125/18 # calculations explained in the docstring

    grad_loss = gradient_loss(pred, target) # what the gradient loss function computes

    assert abs(grad_loss.item() - expected_loss) < 1e-6, (
        f"gradient_loss should GIVE the handy calculated value {expected_loss}, got {grad_loss.item()}"
    )