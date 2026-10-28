
import torch
from unet import ResBlock, UNet
import torch.nn.functional as F

def test_ResBlock_output_size():
    """
    GIVEN: an input with in_channels channels
    WHEN: ResBlock forward is called
    THEN: the output has the out_channels channels and the same spatial size (B, C, H, W)
    """
    torch.manual_seed(42) # for reproduciblity across runs
    block = ResBlock(in_channels=2, out_channels=32) # instantiate the Resblock (building block of the UNet architecture)

    output = block(torch.randn(8, 2, 300, 300)) # forward a random input through the block

    assert output.shape == (8, 32, 300, 300), f"Expected shape (8, 32, 300, 300), got {tuple(output.shape)}"
    
    
def test_unet_output_is_non_negative():
    """
    GIVEN: an input that contains negative values
    WHEN: UNet forward is called
    THEN: every output value is >= 0, as projection intensities cannot be negative
    """
    torch.manual_seed(42)  # for reproduciblity across runs
    model = UNet(in_channels=2, out_channels=2)  # instantiate the UNet
    model.eval()  # switch to evaluation mode (do inference) and no training mode

    with torch.no_grad():  # here we only do inference, so no gradients needed
        output = model(torch.randn(8, 2, 300, 300))  # forward a random input through the model

    
    assert torch.all(output >= 0), "The output should be non-negative (it is clamped by design)"


def test_unet_keeps_output_shape_divisible_hw():
    """
    GIVEN: an input with shape whose height and width are divisible by 16 e.g. (8, 2, 160, 160)
                 all height and width values during halving with MaxPool2d(2) in the UNet are integer
    WHEN: UNet forward is called
    THEN: the output has exactly the same shape as input. 
    """
    
    torch.manual_seed(42)  # for reproducible across runs
    model = UNet(in_channels=2, out_channels=2)  # instantiate the UNet
    model.eval()  # switch to evaluation mode (do inference) and no training mode
    
    # do inference, no gradients needed
    with torch.no_grad():  
        output = model(torch.randn(8, 2, 160, 160))  # forward a random input through the model

    assert output.shape == (8, 2, 160, 160), f"Expected shape (8, 2, 160, 160), got {tuple(output.shape)}"
    
   
def test_unet_keeps_output_shape_nondivisible_hw():
    """
    GIVEN: an input with shape whose height and width are not divisible by 16 e.g. (8, 2, 301, 301)
                height and width values during halving with MaxPool2d(2) in the UNet become decimal number
    WHEN: UNet forward is called
    THEN: the output has exactly the same shape as input. 
               301 is not divisible by 16, so this also checks that the decoder realigns its features
               with the skip connections.
    """
    
    torch.manual_seed(42)  # for reproducible across runs
    model = UNet(in_channels=2, out_channels=2)  # instantiate the UNet
    model.eval()  # switch to evaluation mode (do inference) and no training mode
    
    # do inference, no gradients needed
    with torch.no_grad():  
        output = model(torch.randn(8, 2, 301, 301))  # forward a random input through the model

    assert output.shape == (8, 2, 301, 301), f"Expected shape (8, 2, 301, 301), got {tuple(output.shape)}"
  
  
"""
====================================================================================================
Tests for _match_size function of UNet class in the "unet.py" 
====================================================================================================
"""

def test_match_size_returns_sameshape_when_equalsize():
    """
    GIVEN: an upsampled tensor and an encoder tensor with the same spatial size
    WHEN: _match_size is called
    THEN: the shape of returned tensor is unchanged
    """
    ups = torch.randn(8, 32, 150, 150)
    encd = torch.randn(8, 32, 150, 150)

    matched_size = UNet._match_size(ups, encd)

    assert matched_size.shape == ups.shape, (
    f"The shape of output tensor of _match_size should remain unchanged when the spatial dimensions already match, " 
    f"Expected shape {tuple(ups.shape)}, got {tuple(matched_size.shape)}"
    )

def test_match_size_returns_samevalues_when_equalsize():
    """
    GIVEN: an upsampled tensor and an encoder tensor with the same spatial size
    WHEN: _match_size is called
    THEN: the returned tensor is unchanged (no interpolation happened)
    """
    
    ups = torch.randn(8, 32, 150, 150)
    encd = torch.randn(8, 32, 150, 150)

    matched_size = UNet._match_size(ups, encd)

    assert torch.equal(matched_size, ups), "The output values of _match_size should be identical to the input values when resizing is not needed."


def test_match_size_ups_smaller():
    """
    GIVEN: an upsampled tensor with smaller spatial dimensions than the encoder tensor
    WHEN: _match_size is called
    THEN: the output with adjusted size has the same height and width as the encoder tensor
                 while batch size and number of channels are not changed.
    """
    
    ups = torch.randn(8, 16, 31, 31)
    encd = torch.randn(8, 16, 32, 32)

    matched_size = UNet._match_size(ups, encd)

    assert matched_size.shape == (8, 16, 32, 32), (
    f"The height and width of output tensor of _match_size should match the encoder tensor's ones, "
    f"Expected shape (8, 16, 32, 32), got {tuple(matched_size.shape)}"
    )
    
def test_match_size_ups_larger():
    """
    GIVEN: an upsampled tensor with larger spatial dimensions than the encoder tensor
    WHEN: _match_size is called
    THEN: the output with matched size has the same height and width as the encoder tensor
                 while batch size and number of channels are not changed.
    """
    
    ups = torch.randn(8, 16, 33, 33)
    encd = torch.randn(8, 16, 32, 32)

    matched_size = UNet._match_size(ups, encd)
    
    assert matched_size.shape == (8, 16, 32, 32), (
    f"The height and width of output tensor of _match_size should match the encoder tensor's ones, "
    f"Expected shape (8, 16, 32, 32), got {tuple(matched_size.shape)}"
    )

def test_match_size_with_different_h_w():
    """
    GIVEN: an upsampled tensor whose height and width differ, and differ drastically with encoder ones
    WHEN: _match_size is called
    THEN: Both spatial dimensions are resized to match the encoder tensor
    """
    
    ups = torch.randn(8, 16, 31, 41)
    encd = torch.randn(8, 16, 39, 53)

    matched_size = UNet._match_size(ups, encd)

    assert matched_size.shape == (8, 16, 39, 53), (
    f"The height and width of output tensor of _match_size should match the encoder tensor's ones, "
    f"Expected shape (8, 16, 39, 53), got {tuple(matched_size.shape)}"
    )


def test_match_size_preserves_dtype():
    """
    GIVEN: an upsampled tensor with a specific data type
    WHEN: _match_size is called to resize its spatial dimensions
    THEN: the output after matching size preserves the input tensor's data type
    """
    ups = torch.randn(8, 16, 31, 31, dtype=torch.float32)
    encd = torch.randn(8, 16, 32, 32, dtype=torch.float32)

    matched_size = UNet._match_size(ups, encd)

    assert matched_size.dtype == ups.dtype, (
    f"The output tensor of _match_size should preserve the data type of the input upsampled tensor, "
    f"Expected dtype {ups.dtype}, got {result.dtype}"
    )
    
def test_match_size_preserves_device():
    """
    GIVEN: an upsampled tensor with a specific device
    WHEN: _match_size is called to resize its spatial dimensions
    THEN: the output after matching size preserves the input tensor's device (cpu or cuda or ...)
    """
    ups = torch.randn(8, 16, 31, 31)
    encd = torch.randn(8, 16, 32, 32)

    matched_size = UNet._match_size(ups, encd)

    assert matched_size.device == ups.device, (
    f"The output tensor of _match_size should remain on the same device as the input upsampled tensor, "
    f"Expected device {ups.device}, got {result.device}"
    )
    
    
def test_match_size_nochange_constant_tensor():
    """
    GIVEN: a tensor whose intensity values are constant value of 0.5
    WHEN: _match_size is called to resize its spatial dimensions
    THEN: the output after matching size should be a tensor with the same constant intensity value
    """
    
    ups  = torch.full((8, 16, 33, 33), 0.5)
    encd = torch.randn(8, 16, 32, 32)

    out = UNet._match_size(ups, encd)

    assert torch.allclose(out, torch.full_like(out, 0.5)), (
    f"Expected all output values of _match_size remain 0.5, "
    f"but got minimum {out.min().item():.4f} and maximum {out.max().item():.4f}"
)