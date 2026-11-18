import math
import pytest
import torch
from libs.dataset import find_paired_files, load_as_tensor
from libs.losses import denoising_loss, gradient_loss
from libs.ssim import gaussian_window, ssim_loss
import os

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

# =================================================
# tests for ssim_loss and gaussian_window function
# =================================================

def test_gaussian_window_shape():
    """
    GIVEN: a window size and a sigma
    WHEN: gaussian_window is called
    THEN: the window has shape (1, 1, K, K)
    """
    window = gaussian_window(window_size=7, sigma=1.5)

    assert window.shape == (1, 1, 7, 7), f"Expected shape (1, 1, 7, 7), got {tuple(window.shape)}"

def test_gaussian_window_shape_and_normalization():
    """
    GIVEN: a window size and a sigma
    WHEN: gaussian_window is called
    THEN: the window has sums to 1, so it computes a
          weighted average
    """
    window = gaussian_window(window_size=7, sigma=1.5)

    assert torch.isclose(window.sum(), torch.tensor(1.0), atol=1e-6), \
        f"Expected the window to sum to 1, got {window.sum().item()}"

def test_gaussian_window_peaks_at_the_centre():
    """
    GIVEN: a window_size and a sigma to create a Gaussian window
    WHEN: gaussian_window is created
    THEN: the centre of the window holds the largest weight (the window has Gaussian shape)
    """
    window = gaussian_window(window_size=5, sigma=1.5)[0, 0]

    assert window[2, 2] == window.max(), "The centre of the window should hold the largest weight"


def test_gaussian_window_is_symmetric():
    """
    GIVEN: a window_size and a sigma to create a Gaussian window
    WHEN: gaussian_window is created
    THEN: the window is symmetric
    """
    window = gaussian_window(window_size=5, sigma=1.5)[0, 0]

    assert torch.allclose(window, window.flip(0)), "The window should be symmetric along the height"
    assert torch.allclose(window, window.flip(1)), "The window should be symmetric along the width"

def test_ssim_loss_gives_scalar_not_map():
    """
    GIVEN: two identical image batches
    WHEN: ssim_loss is computed
    THEN: the loss is ~0 (SSIM of an image with itself is 1)
    """
    torch.manual_seed(42)

    prediction = torch.rand(4, 2, 300, 300)

    target = torch.rand(4, 2, 300, 300)

    ssim_loss_value = ssim_loss(prediction, target)

    assert ssim_loss_value.dim() == 0, f"Expected a scalar loss, got shape {tuple(ssim_loss_value.shape)}"

def test_ssim_loss_is_zero_for_identical_images():
    """
    GIVEN: two identical projection batches
    WHEN: ssim_loss is computed
    THEN: the loss is ~0 (SSIM of an projection with itself is 1)
    """
    torch.manual_seed(42)
    projections = torch.rand(4, 2, 300, 300)

    ssim_loss_value = ssim_loss(projections, projections)

    assert torch.isclose(ssim_loss_value, torch.tensor(0.0), atol=1e-5), \
        f"Expected ~0 for identical images, got {ssim_loss_value.item()}"

def test_ssim_loss_swap_prediction_target():
    """
    GIVEN: two different images
    WHEN: ssim_loss is computed in both orders
    THEN: the ssim loss is the same regardless of order of entries 
          because the SSIM formula is symmetric in its two entries.
    """

    torch.manual_seed(42)
    prediction = torch.rand(1, 2, 24, 24)
    
    target = torch.rand(1, 2, 24, 24)

    ssim_loss_value1 =ssim_loss(prediction, target)
    ssim_loss_value2 = ssim_loss(target, prediction)

    assert torch.isclose(ssim_loss_value1, ssim_loss_value2, atol=1e-6), \
        "ssim_loss should not depend on the order of entries"

def test_ssim_loss_grows_with_noise():
    """
    GIVEN: a clean projection batch as target and
          two noisy versions of it, one noisier than the other
    WHEN: ssim_loss is computed between the target and each noisy version
    THEN: the noisier prediction batch gives the larger ssim loss
    """
    torch.manual_seed(42)
    noiseless_target = torch.rand(4, 2, 300, 300)
    noise = torch.randn(4, 2, 300, 300)

    low_noise_pred = (noiseless_target + 0.09 * noise)
    higher_noise_pred = (noiseless_target + 0.1 * noise)

    low_noise_loss = ssim_loss(low_noise_pred, noiseless_target)
    higher_noise_loss = ssim_loss(higher_noise_pred, noiseless_target)

    assert low_noise_loss < higher_noise_loss, (
        f"A noisier prediction should give a larger loss, got {low_noise_loss.item()} "
        f"(small noise) vs {higher_noise_loss.item()} (large noise)"
    )

def test_ssim_loss_detects_a_structural_difference():
    """
    GIVEN: two projections with the same mean equal to 0.5 (SSIM depends on mean) but a different structure
           (vertical vs horizontal stripes)
    WHEN: ssim_loss is computed
    THEN: the loss is clearly positive, i.e. SSIM sees more than the mean value
    """

    vertical_stips_proj = torch.zeros(1, 1, 300, 300)
    vertical_stips_proj[:, :, :, ::2] = 1.0 # every other column is 1, the rest are 0, Hence the mean is 0.5

    horizontal_stips_proj = torch.zeros(1, 1, 300, 300)
    horizontal_stips_proj[:, :, ::2, :] = 1.0 # every other row is 1, the rest are 0, Hence the mean is 0.5

    ssim_loss_value = ssim_loss(vertical_stips_proj, horizontal_stips_proj)

    assert ssim_loss_value.item() > 0.99, f"Expected a large loss for different structures, got {ssim_loss_value.item()}"


def test_ssim_loss_matches_handy_calculated_value():
    """
    GIVEN: a 2x2 image `pred = [[1, 0], [0, 1]]` and a target `0.5 * pred`
    WHEN: ssim_loss is computed with a 3x3 Gaussian window (sigma = 1.5)
    THEN: the loss equals the value calculated by hand below

    How SSIM is calculated by hand:
      1. The 3x3 window weights are w = [a, b, a] / sum, with
         g = exp(-coord^2 / (2 * sigma^2)) 
            for coord = arange(3)-3//2 = [0, 1, 2] - 1 = -1, 0, 1, 
            so, a = g(coord=1) and b = g(coord=0) = 1,
            normalized to sum to 1. 
            This gives a = 0.30780 and b = 0.38440. 
         
         The 2D weight of a neighbour is the product of the 1D weights: centre b*b, side a*b, corner a*a.
      
      
      2. In a 2x2 image every 3x3 window (zero-padded) covers all 4 pixels, so each pixel's
         local statistics are weighted sums over the whole image:

         since the prediction is binary (0 or 1), the target is half of prediction, we 
         calculate the local means, variances and covariance according to prediction variables i.e. mu_p, var_p

            mu_p   = sum(w * p)          
            mu_t = 0.5 * mu_p   (since t = 0.5 * p)
            var_p  = sum(w * p^2) - mu_p^2    
            var_t = sum(w * t^2) - mu_t^2 = 0.25 * var_p
            cov = sum(w * p * t) - mu_p * mu_t = 0.5 * var_p

            prediction:

                1       0

                0       1

            target:

                0.5      0

                0      0.5
                
            weighted by the 3x3 Gaussian window (sigma = 1.5):

                a.a    a.b     a.a

                a.b    b.b     a.b

                a.a    a.b     a.a

        SSIM = (2*mu_p*mu_t + C1) * (2*cov + C2) / 
            ((mu_p^2 + mu_t^2 + C1) * (var_p + var_t + C2))
    
    since t = 0.5 * p :

        SSIM = (mu_p^2 + C1) * (var_p + C2) / 
               ((1.25*mu_p^2 + C1) * (1.25*var_p + C2))

           
    3. The ones of `pred` sit on the diagonal, the rest are zeros,
        the 1st and 3rd convolutions are similar 
         and 
        the 2nd and 4th convolutions are similar, 

    1st and 3rd convolution:

        1st:
         - pixel (0,0) = 1 
           weight b*b
         
         - pixel  (1,1) = 1: 
           weight a*a 
        
        3rd:
         - pixel (0,0) = 1 
           weight b*b
         
         - pixel  (1,1) = 1: 
           weight a*a 

         - for both convolutions,the two pixels (0,1) and (1,0) are 0:
           weights a*b each

         - the local mean and variance for each pixel:
                mu_p = b*b + a*a = 0.38440*0.38440 + 0.30780*0.30780 = 0.24250 
                mu_t = 0.5 * mu_p = 0.12125

                var_p = sum(w * p^2) - mu_p^2 => (p is binary, p^2 = p)
                sum (w * p*2) = sum(w*p) = 0.24250
                        mu_p - mu_p^2 = 0.24250 - 0.24250^2 = 0.18370
                    var_t = 0.25 * var_p = 0.045925
                for 1st and 3rd convolution these calculation is the same, Hence:
                      SSIM = 0.6408636
        
        -2nd:
            - pixel (0,1)=0:
            weight a*b

            - pixel (1,0)=0:
            weight a*b

        -4th:
            - pixel (1,0)=0:
            weight a*b

            - pixel (0,1)=0:
            weight a*b

        -for both convolutions the two pixels (0,1) and (1,0) are 1:
         weight a*b each

        -the local mean and variance for each pixel:
                mu_p = 2*a*b = 0.23664
                mu_t = 0.5 * mu_p = 0.11832      
                var_p = same as above = mu_p - mu_p^2 = 0.18064
                var_t = 0.25 * var_p = 0.04516

                for 2nd and 4th convolution these calculation is the same, Hence:
                       SSIM = 0.640842  


        mean(SSIM) = 1st conv + 2nd conv + 3rd conv + 4th conv / 4 
                    = (0.6408636 + 0.6408422 + 0.6408636 + 0.6408422)/4 = 0.640852

    
      5. The loss is 1 - mean(SSIM) = 1 - 0.640852 = 0.359148
    """


    pred = torch.tensor([[1., 0.],
                         [0., 1.]]).view(1, 1, 2, 2)
    target = 0.5 * pred

    loss = ssim_loss(pred, target, window_size=3)

    expected_loss = 0.359148
    assert abs(loss.item() - expected_loss) < 1e-6, (
        f"ssim_loss should match the handy calculated value {expected_loss}, got {loss.item()}"
    )


def test_ssim_loss_returns_values_in_valid_range():
    """
    GIVEN: a target in the range of [0, 1] and 
           a prediction with values above 1 and below 0, as an untrained model
           can produce in the first epochs of training.
    WHEN: ssim_loss is computed
    THEN: the result is in the range [0, 1]
    """
    torch.manual_seed(42)
    pred = torch.randn(1, 2, 16, 16) * 2 # normal distribution with mean 0, std 2,
                                         # so values outside range [-2,2] are likely

    target = torch.rand(1, 2, 16, 16) # uniform distribution strictly in the range [0, 1]

    ssim_loss_value = ssim_loss(pred, target)

    assert 0 <= ssim_loss_value <= 1, \
        "SSIM loss values must be in the range [0, 1]"


def test_ssim_loss_gives_zero_for_perfect_prediction():
    """
    GIVEN: a prediction whose both channels are perfectly predicted
    WHEN: ssim_loss is computed on both channels together and separately
    THEN: the loss of each channel and the loss of both channels together are all ~0
    """

    # create a target with two channels, each filled with a constant value
    target_ch0 = torch.full((4, 1, 300, 300), 0.5)
    target_ch1 = torch.full((4, 1, 300, 300), 0.4)

    # concatenate the two channels to form a target tensor of shape (4, 2, 300, 300)
    target = torch.cat([target_ch0, target_ch1], dim=1)

    # create a prediction with two channels, each filled with the same constant value as the target
    pred_ch0 = torch.full((4, 1, 300, 300), 0.5)
    pred_ch1 = torch.full((4, 1, 300, 300), 0.4)

    # concatenate the two channels to form a prediction tensor of shape (4, 2, 300, 300)
    pred = torch.cat([pred_ch0, pred_ch1], dim=1)

    # calculate the ssim loss for both channels together and each channel separately
    both = ssim_loss(pred, target)
    first = ssim_loss(pred_ch0, target_ch0)
    second = ssim_loss(pred_ch1, target_ch1)

    assert abs(first.item()) < 1e-6, f"The first channel (perfectly predicted) should have ~0 loss, got {first.item()}"
    assert abs(second.item()) < 1e-6, f"The second channel (perfectly predicted) should have ~0 loss, got {second.item()}"
    assert abs(both.item()) < 1e-6, f"The both channels (perfectly predicted) should have ~0 loss, got {both.item()}"
    
@pytest.mark.parametrize("inp, tgt", ALL_PROJECTION_PAIRS)
def test_ssim_loss_treats_channels_independently(inp, tgt):
    """
    GIVEN: mouse projection pair (pred, target) whose shape is (2, 300, 300)
    WHEN: ssim_loss is computed on both channels together and each channel separately
    THEN: the returned ssim loss equals the mean of the two single-channel ssim losses
    """

    # reshape the input and target tensors to have a batch dimension of 1
    pred = inp.unsqueeze(0)     # change (2, 300, 300) to (1, 2, 300, 300)
    target = tgt.unsqueeze(0)   # change (2, 300, 300) to (1, 2, 300, 300)

    # separate the channels of the prediction and target tensors
    pred_ch0 =pred[:, 0:1]
    pred_ch1 = pred[:, 1:2]
    target_ch0 = target[:, 0:1]
    target_ch1 = target[:, 1:2]

    # calculate the ssim loss for both channels together and each channel separately
    both = ssim_loss(pred, target)
    first = ssim_loss(pred_ch0, target_ch0)
    second = ssim_loss(pred_ch1, target_ch1)

    assert torch.isclose(both, (first + second) / 2, atol=1e-6), (
        f"Expected mean of channel losses {((first + second) / 2).item():.6f}, "
        f"got {both.item():.6f}"
    )

def test_ssim_loss_handles_small_differences():
    """
    GIVEN: a prediction whose channels have small differences from the target channels
    WHEN: ssim_loss is computed on both channels together and separately
    THEN: the loss of each channel and the loss of both channels are computed correctly, 
          so, stability constants are working correctly.
    """

    # create a target with two channels, each filled with a constant value
    target_ch0 = torch.full((4, 1, 300, 300), 0.5)
    target_ch1 = torch.full((4, 1, 300, 300), 0.4)
    
    # concatenate the two channels to form a target tensor of shape (4, 2, 300, 300)
    target = torch.cat([target_ch0, target_ch1], dim=1)

    # create a prediction with two channels, each filled with ...
    # constant values slightly different than target.
    pred_ch0 = torch.full((4, 1, 300, 300), 0.5*1e-6)  # almost perfect prediction for channel 0
    pred_ch1 = torch.full((4, 1, 300, 300), 0.4*1e-6)  # almost perfect prediction for channel 1
    
    # concatenate the two channels to form a prediction tensor of shape (4, 2, 300, 300)
    pred = torch.cat([pred_ch0, pred_ch1], dim=1) 

    # calculate the ssim loss for both channels together and each channel separately
    both = ssim_loss(pred, target)
    first = ssim_loss(pred_ch0, target_ch0)
    second = ssim_loss(pred_ch1, target_ch1)

    assert torch.isclose(both, (first + second) / 2, atol=1e-8), \
        f"Expected the mean of the channel losses {((first + second) / 2).item()}, got {both.item()}"

def test_ssim_loss_handles_large_differences():
    """
    GIVEN: a prediction whose channels have large differences from the target channels
    WHEN: ssim_loss is computed on both channels together and separately
    THEN: the loss of each channel and the loss of both channels are computed correctly, 
          so, stability constants are working correctly.
    """

    # create a target with two channels, each filled with a constant value
    target_ch0 = torch.full((4, 1, 300, 300), 0.0)
    target_ch1 = torch.full((4, 1, 300, 300), 0.0)
    
    # concatenate the two channels to form a target tensor of shape (4, 2, 300, 300)
    target = torch.cat([target_ch0, target_ch1], dim=1)

    # create a prediction with two channels, each filled with ...
    # constant values slightly different than target.
    pred_ch0 = torch.full((4, 1, 300, 300), 1.0)  # almost perfect prediction for channel 0
    pred_ch1 = torch.full((4, 1, 300, 300), 1.0)  # almost perfect prediction for channel 1
    
    # concatenate the two channels to form a prediction tensor of shape (4, 2, 300, 300)
    pred = torch.cat([pred_ch0, pred_ch1], dim=1) 

    # calculate the ssim loss for both channels together and each channel separately
    both = ssim_loss(pred, target)
    first = ssim_loss(pred_ch0, target_ch0)
    second = ssim_loss(pred_ch1, target_ch1)

    assert torch.isclose(both, (first + second) / 2, atol=1e-8), \
        f"Expected the mean of the channel losses {((first + second) / 2).item()}, got {both.item()}"

@pytest.mark.parametrize("inp, tgt", ALL_PROJECTION_PAIRS)
def test_ssim_loss_backpropagates_gradient(inp, tgt):
    """
    GIVEN: a prediction that requires a gradient
    WHEN: ssim_loss is back-propagated
    THEN: a gradient reaches the prediction, so the term can
          actually train the model
    """

    # reshape the input and target tensors to have a batch dimension of 1
    pred = inp.unsqueeze(0)     # change (2, 300, 300) to (1, 2, 300, 300)
    target = tgt.unsqueeze(0)   # change (2, 300, 300) to (1, 2, 300, 300)

    # ensure the prediction requires a gradient
    pred.requires_grad_(True)
    
    # compute the ssim loss and backpropagate to get the gradient
    ssim_loss(pred, target).backward()

    assert pred.grad is not None, "No gradient reached the prediction, " \
        "which means gradients cannot flow back to the model and training fails"

@pytest.mark.parametrize("inp, tgt", ALL_PROJECTION_PAIRS)
def test_ssim_loss_gradient_is_finite(inp, tgt):
    """
    GIVEN: a prediction that requires a gradient
    WHEN: ssim_loss is back-propagated
    THEN: a finite gradient reaches the prediction
    """

    # reshape the input and target tensors to have a batch dimension of 1
    pred = inp.unsqueeze(0)     # change (2, 300, 300) to (1, 2, 300, 300)
    target = tgt.unsqueeze(0)   # change (2, 300, 300) to (1, 2, 300, 300)

    # ensure the prediction requires a gradient
    pred.requires_grad_(True)

    # compute the ssim loss and backpropagate to get the gradient
    ssim_loss(pred, target).backward()

    assert torch.isfinite(pred.grad).all(), "The gradient contains NaN or infinite values"


@pytest.mark.parametrize("inp, tgt", ALL_PROJECTION_PAIRS)
def test_ssim_loss_gradient_is_nonzero(inp, tgt):
    """
    GIVEN: a prediction that requires a gradient
    WHEN: ssim_loss is back-propagated
    THEN: a non-zero gradient reaches the prediction
    """

    # reshape the input and target tensors to have a batch dimension of 1
    pred = inp.unsqueeze(0)     # change (2, 300, 300) to (1, 2, 300, 300)
    target = tgt.unsqueeze(0)   # change (2, 300, 300) to (1, 2, 300, 300)

    # ensure the prediction requires a gradient
    pred.requires_grad_(True)

    # compute the ssim loss and backpropagate to get the gradient
    ssim_loss(pred, target).backward()

    assert pred.grad.abs().sum() > 0, "The gradient is zero everywhere"

@pytest.mark.parametrize("inp, tgt", ALL_PROJECTION_PAIRS)
def test_ssim_loss_decreases_with_gradient_step(inp, tgt):
    torch.manual_seed(0)

    # reshape the input and target tensors to have a batch dimension of 1
    pred = inp.unsqueeze(0)     # change (2, 300, 300) to (1, 2, 300, 300)
    target = tgt.unsqueeze(0)   # change (2, 300, 300) to (1, 2, 300, 300)

    # ensure the prediction requires a gradient
    pred.requires_grad_(True)

    loss1 = ssim_loss(pred, target) # compute the initial loss
    loss1.backward() # compute the gradient of the loss with respect to the prediction

    # a simulation of a gradient descent step, which should reduce the loss
    with torch.no_grad(): 
        pred -= 0.1 * pred.grad # take a small gradient step to reduce the loss
    pred.grad = None # reset the gradient for the next backward pass
    loss2 = ssim_loss(pred, target) # compute the loss after the gradient step

    assert loss2 < loss1, f"Expected loss to decrease after gradient step, \
                but got loss2={loss2.item()} which is larger than loss1={loss1.item()}"