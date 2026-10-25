
import torch
from unet import ResBlock, UNet

def test_projection_ResBlock_changes_channels_and_keeps_size():
    """
    GIVEN: an input with in_channels channels
    WHEN: ResBlock forward is called
    THEN: the output has the out_channels channels and the same spatial size
    """
    torch.manual_seed(42)
    block = ResBlock(in_channels=2, out_channels=32)

    output = block(torch.randn(2, 2, 24, 24))

    assert output.shape == (2, 32, 24, 24), f"Expected shape (2, 32, 24, 24), got {tuple(output.shape)}"


def test_unet_output_is_non_negative():
    """
    GIVEN: an input that contains negative values
    WHEN: UNet forward is called
    THEN: every output value is >= 0, as projection intensities cannot be negative
    """
    torch.manual_seed(42)
    model = UNet(in_channels=2, out_channels=2)
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(1, 2, 300, 300))

    assert torch.all(output >= 0), "The output should be non-negative (it is clamped by design)"

