
import torch
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
    coord = torch.arange(window_size, dtype=torch.float32) - window_size // 2 # coord: x or y 

    # Gaussian function: exp(-x^2 / (2 * sigma^2))
    gauss_func = torch.exp(-(coord ** 2) / (2 * sigma ** 2))

    # Create the 2D Gaussian window (kernel) by taking the outer product of the 1D kernel with itself
    window_2D = gauss_func[:, None] * gauss_func[None, :]  # outer product: G(y) * G(x)

    # Reshape the window to (1, 1, window_size, window_size) for 2D convolution of a (B, C, H, W)
    window = window_2D.view(1, 1, window_size, window_size)

    return window


def ssim_loss(pred, target, window_size=5, sigma=1.5):
    """
    SSIM loss (1 - SSIM) between batches of
    inputs projections and targets projections, each of shape (B, C, H, W).

    Args:
        pred: predicted projections, shape (B, C, H, W).
        target: noiseless target projections, shape (B, C, H, W).
        window_size: Side length of the Gaussian window.
        sigma: Standard deviation of the Gaussian window.

    The formula of ssim function is:
    SSIM(x, y) = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2) / 
                 ((mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2))
 
    Returns:
        Scalar ssim loss
    """

    # Stability constants from the original SSIM paper (K1 = 0.01, K2 = 0.03),
    # squared and for a data range of [0, 1].
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    height, width = pred.shape[2:]

    # Compare one channel at a time
    pred = pred.reshape(-1, 1, height, width)
    target = target.reshape(-1, 1, height, width)
    
    window = gaussian_window(window_size, sigma)
    padding = window_size // 2

    # match the dtype and device of the window to the predicted tensor
    window = gaussian_window(window_size, sigma).to(pred)

    # Local means
    mu_pred = F.conv2d(pred, window, padding=padding)
    mu_target = F.conv2d(target, window, padding=padding)

    # Local variances and covariance
    var_pred = F.conv2d(pred * pred, window, padding=padding) - mu_pred ** 2
    var_target = F.conv2d(target * target, window, padding=padding) - mu_target ** 2
    covariance = F.conv2d(pred * target, window, padding=padding) - mu_pred * mu_target

    ssim_map = ((2 * mu_pred * mu_target + C1) * (2 * covariance + C2) /
                ((mu_pred ** 2 + mu_target ** 2 + C1) * (var_pred + var_target + C2)))
    
    # ssim loss is 1 - ssim value
    ssim_loss_value = 1 - ssim_map.mean()
    return ssim_loss_value
