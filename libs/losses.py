
import torch.nn.functional as F
from libs.ssim import ssim_loss


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


def denoising_loss(outputs, targets, mae_weight=1, ssim_weight=1, gradient_weight=1):
    """
    Combined denoising loss, computed on both channels (LE and HE).

    Args:
        outputs: Predicted (denoised) projections, shape (B, C, H, W).
        targets: target (noiseless) projections, shape (B, C, H, W).
        mae_weight: Weight of the mean absolute error term.
        ssim_weight: Weight of the SSIM term with a 3x3 window.
        gradient_weight: Weight of the gradient term.

    Returns:
        one scalar loss, which is the weighted sum of the three components.
    """

    loss = (mae_weight*F.l1_loss(outputs, targets) + ssim_weight*ssim_loss(outputs, targets, window_size=3) + gradient_weight*gradient_loss(outputs, targets))


    return loss