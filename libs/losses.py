
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_window(window_size, sigma=1.5):
    """
    Build the normalized 2D Gaussian window that weights each neighbourhood.

    Args:
        window_size: Side length of the square window (e.g. 3 or 5).
        sigma: Standard deviation of the Gaussian.

    Returns:
        Tensor of shape (1, 1, window_size, window_size), summing to 1.
    """

    # Create a 1D Gaussian kernel
    # The coordinates are centered around 0, 
    # e.g. for a window of size 5, the coordinates are [-2, -1, 0, 1, 2].
    coord = torch.arange(window_size, dtype=torch.float32) - window_size // 2 # coord: x or y 

    # Gaussian function: exp(-x^2 / (2 * sigma^2))
    gauss_func = torch.exp(-(coord ** 2) / (2 * sigma ** 2))

    # Create the 2D Gaussian window (kernel) by taking the outer product of the 1D kernel with itself
    window_2D = gauss_func[:, None] * gauss_func[None, :]  # outer product: G(y) * G(x)

    # Reshape the window to (1, 1, window_size, window_size) for 2D convolution of a (B, C, H, W)
    # because window is convolved with each channel and each projection separately
    # so we need to add two singleton dimensions for batch and channel.
    window = window_2D.view(1, 1, window_size, window_size)

    window_normed = window / window.sum()

    return window_normed


class SSIMLoss(nn.Module):
    """
    SSIM loss (1 - SSIM) between batches of predicted projections and target projections,
    each of shape (B, C, H, W).

    The Gaussian window is built once when the object is created and is kept as a non trainable and
    not saved in the state dict buffer of the module, so it follows the module with ".to(device/dtype)".

    The data range (L) is fixed to 1, since the projections are normalized to [0, 1].

    The formula of ssim function is:
    SSIM(x, y) = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2) / 
                 ((mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2))

    it is composed of multimplication of 3 terms: luminance * contrast * structure, where:

    the luminance term: (2 * mu_x * mu_y + C1) / (mu_x^2 + mu_y^2 + C1)
    the contrast term: (2 * sigma_x * sigma_y + C2) / (sigma_x^2 + sigma_y^2 + C2)
    the structure term: (sigma_xy + C3) / (sigma_x * sigma_y + C3), where C3 = C2 / 2   

    Where:
            mu_x: local mean of x
            mu_y: local mean of y
            sigma_x^2: local variance of x
            sigma_y^2: local variance of y
            sigma_xy: local covariance of x and y            
                 
    Args:
        window_size: Side length of the Gaussian window.
        sigma: Standard deviation of the Gaussian window.

    Example:
        >>> ssim_loss = SSIMLoss(window_size=3)
        >>> loss = ssim_loss(prediction, target)   # scalar tensor
    """

    # Stability constants from the original SSIM paper (K1 = 0.01, K2 = 0.03)
    K1 = 0.01
    K2 = 0.03
    DATA_RANGE = 1

    def __init__(self, window_size=5, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma

        # persistent=False: the window is a constant, not a learnable/saved parameter.
        self.register_buffer('window', gaussian_window(window_size, sigma), persistent=False)

    def forward(self, pred, target):
        """
        Args:
            pred: predicted projections, shape (B, C, H, W).
            target: noiseless target projections, shape (B, C, H, W).

        Each channel is compared with the matching channel of the other image
        (LE with LE, HE with HE) by folding the channels into the batch
        dimension, and the result is averaged over all pixels and channels.
 
        ssim value closer to 1, means two projections are more similar, means loss is closer to 0
        Hence, ssim loss = 1 - ssim (0 when the two batches are identical)

        Returns:
                Scalar ssim loss, which is the average of the ssim values over all pixels and channels.
        """

        C1 = (self.K1 * self.DATA_RANGE) ** 2
        C2 = (self.K2 * self.DATA_RANGE) ** 2

        B, C, height, width = pred.shape

        # Fold channels into the batch dim so each channel is convolved independently.
        # which matchs the window shape (1, 1, window_size, window_size)
        pred_flat = pred.reshape(B * C, 1, height, width)
        target_flat = target.reshape(B * C, 1, height, width)

        # match the dtype and device of the window to the predicted tensor
        window = self.window.to(pred_flat)

        padding = self.window_size // 2

        # Local means
        mu_pred = F.conv2d(pred_flat, window, padding=padding)
        mu_target = F.conv2d(target_flat, window, padding=padding)

        # Local variances and covariance
        var_pred = F.conv2d(pred_flat * pred_flat, window, padding=padding) - mu_pred ** 2
        var_target = F.conv2d(target_flat * target_flat, window, padding=padding) - mu_target ** 2
        covariance = F.conv2d(pred_flat * target_flat, window, padding=padding) - mu_pred * mu_target


        # SSIM, as defined in the paper: luminance * contrast * structure
        # after multiplying, we reach to the following formula:
        ssim_map = ((2 * mu_pred * mu_target + C1) * (2 * covariance + C2) /
                    ((mu_pred ** 2 + mu_target ** 2 + C1) * (var_pred + var_target + C2)))
        
        # more ssim value means more similarity, means less loss of data, hence loss is 1 - ssim
        return 1 - ssim_map.mean()


def gradient_loss(pred, target):
    """
    Edge-preserving loss: mean squared error between gradients of
    the predicted and target images.

    The gradients are differences between neighbouring pixels, along
    the height and along the width.

    Args:
        pred: Predicted images, shape (B, C, H, W). B: batch size, C: number of channels
        target: Ground truth images, shape (B, C, H, W).

    Returns:
        Scalar loss: the average of the height and width gradient MSEs.
    """
    dh_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dh_target = target[:, :, 1:, :] - target[:, :, :-1, :]

    dw_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dw_target = target[:, :, :, 1:] - target[:, :, :, :-1]

    gradient_loss_value =(F.mse_loss(dh_pred, dh_target) + F.mse_loss(dw_pred, dw_target)) / 2

    return gradient_loss_value


def denoising_loss(predictions, targets, mae_weight=1, ssim_weight=1, gradient_weight=1):
    """
    Combined denoising loss, computed on both channels (LE and HE).

    Args:
        prediction: Predicted (denoised) projections, shape (B, C, H, W).
        targets: target (noiseless) projections, shape (B, C, H, W).
        mae_weight: Weight of the mean absolute error term.
        ssim_weight: Weight of the SSIM term with a 3x3 window.
        gradient_weight: Weight of the gradient term.

    Returns:
        one scalar loss, which is the weighted sum of the three components.
    """

    ssim_loss = SSIMLoss(window_size=3)

    loss = (mae_weight*F.l1_loss(predictions, targets) + ssim_weight*ssim_loss(predictions, targets) + gradient_weight*gradient_loss(predictions, targets))


    return loss
