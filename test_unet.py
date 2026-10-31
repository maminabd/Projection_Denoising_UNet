
import torch
from unet import ResBlock, UNet
import torch.nn.functional as F

def test_ResBlock_output_size():
    """
    GIVEN: an input with in_channels = 2 channels
    WHEN: ResBlock forward is called
    THEN: the output of ResBlock has the out_channels = 32 channels with the same batch_size and spatial size i.e. (B, _, H, W)
    """
    torch.manual_seed(42) # for reproduciblity across runs
    block = ResBlock(in_channels=2, out_channels=32) # instantiate the Resblock (building block of the UNet architecture)

    output = block(torch.randn(8, 2, 300, 300)) # forward a random input through the block

    assert output.shape == (8, 32, 300, 300), f"Expected shape (8, 32, 300, 300) in output of ResBlock, got {tuple(output.shape)}"


def test_ResBlock_output_dtype():
    """
    GIVEN: a float32 input tensor
    WHEN: ResBlock forward is called
    THEN: the output tensor of ResBlock is still float32
    """
    torch.manual_seed(42) # for reproduciblity across runs
    block = ResBlock(in_channels=2, out_channels=32) # instantiate the Resblock (building block of the UNet architecture)

    output = block(torch.randn(8, 2, 300, 300, dtype=torch.float32)) # forward a random float32 input through the block

    assert output.dtype == torch.float32, f"Expected float32 in output of ResBlock, got {output.dtype}"


def test_ResBlock_output_is_non_negative():
    """
    GIVEN: an input that contains negative values
    WHEN: ResBlock forward is called
    THEN: every output value of ResBlock is >= 0, since the block ends with a ReLU
    """
    torch.manual_seed(42) # for reproduciblity across runs
    block = ResBlock(in_channels=2, out_channels=32) # instantiate the Resblock (building block of the UNet architecture)

    output = block(torch.randn(8, 2, 300, 300)) # forward a random input through the block

    assert torch.all(output >= 0), "ResBlock output should be non-negative."


def test_unet_output_is_non_negative():
    """
    GIVEN: an input that contains negative values
    WHEN: UNet forward is called
    THEN: every output value is >= 0, as projection intensities cannot be negative.
    """
    torch.manual_seed(42)  # for reproduciblity across runs
    model = UNet(in_channels=2, out_channels=2)  # instantiate the UNet
    model.eval()  # switch to evaluation mode (do inference) and no training mode

    with torch.no_grad():  # here we only do inference, so no gradients needed
        output = model(torch.randn(8, 2, 300, 300))  # forward a random input through the model

    
    assert torch.all(output >= 0), "The UNet output should be non-negative."


def test_unet_keeps_output_shape_divisible_hw():
    """
    GIVEN: an input with shape whose height and width are divisible by 16 e.g. (8, 2, 160, 160)
                 all height and width values during halving with MaxPool2d(2) in the UNet are integers.
    WHEN: UNet forward is called
    THEN: the output of UNet has exactly the same shape as input. 
    """
    
    torch.manual_seed(42)  # for reproducible across runs
    model = UNet(in_channels=2, out_channels=2)  # instantiate the UNet
    model.eval()  # switch to evaluation mode (do inference) and no training mode
    
    # do inference, no gradients needed
    with torch.no_grad():  
        output = model(torch.randn(8, 2, 160, 160))  # forward a random input through the model

    assert output.shape == (8, 2, 160, 160), f"Expected shape (8, 2, 160, 160) in output of UNet, got {tuple(output.shape)}"
    
   
def test_unet_keeps_output_shape_nondivisible_hw():
    """
    GIVEN: an input with shape whose height and width are not divisible by 16 e.g. (8, 2, 301, 301)
                height and width values during halving with MaxPool2d(2) in the UNet become decimal numbers.
    WHEN: UNet forward is called
    THEN: the output of UNet has exactly the same shape as input. 
    """
    
    torch.manual_seed(42)  # for reproducible across runs
    model = UNet(in_channels=2, out_channels=2)  # instantiate the UNet
    model.eval()  # switch to evaluation mode (do inference) and no training mode
    
    # do inference, no gradients needed
    with torch.no_grad():  
        output = model(torch.randn(8, 2, 301, 301))  # forward a random input through the model

    assert output.shape == (8, 2, 301, 301), f"Expected shape (8, 2, 301, 301) in output of UNet, got {tuple(output.shape)}"
  
def test_unet_respects_out_channels():
    """
    GIVEN: a UNet built with an out_channels value different from in_channels
    WHEN: forward is called
    THEN: the output has exactly out_channels channels
    """
    torch.manual_seed(42) # for reproducible across runs
    model = UNet(in_channels=2, out_channels=1) # instantiate the UNet with 2 channels input and 1 channel output
    model.eval() # switch to evaluation mode (do inference) and no training mode

    # do inference, no gradients needed
    with torch.no_grad():
        output = model(torch.randn(4, 2, 300, 300)) # forward a random 2 channels input through the model

    assert output.shape == (4, 1, 300, 300), f"Expected shape (4, 1, 300, 300) in output of UNet, got {tuple(output.shape)}"


def test_unet_supports_batch_size_one():
    """
    GIVEN: a single-sample batch (B=1) (in case of low GPU memory or for inference on a single image)
    WHEN: UNet forward is called in eval mode
    THEN: the forward pass runs without error and returns the expected shape
    """
    torch.manual_seed(42) # for reproducibility across runs
    model = UNet(in_channels=2, out_channels=2) # instantiate the UNet
    model.eval() # switch to evaluation mode (do inference) and no training mode

    with torch.no_grad():
        output = model(torch.randn(1, 2, 300, 300))

    assert output.shape == (1, 2, 300, 300), f"Expected shape (1, 2, 300, 300) in output of UNet, got {tuple(output.shape)}"


def test_unet_trains_every_parameter():
    """
    GIVEN: a UNet in training mode
    WHEN: a loss computed from its output is backpropagated
    THEN: every parameter in every blockreceives a gradient
    """
    torch.manual_seed(42) # for reproducibility across runs
    model = UNet(in_channels=2, out_channels=2) # instantiate the UNet
    model.train() # switch to training mode (for dropout, batchnorm, etc.)

    output = model(torch.rand(2, 2, 31, 31)) # forward a random input through the model
    output.sum().backward() # compute gradients by backpropagating loss (sum of outputs)

    # named_parameters() addresses (name, parameter) for every trainable weight/bias,
    # e.g. ("down1.attn.query.weight", <Parameter [128,128,1,1]>).
    unused = [name for name, parameter in model.named_parameters() if parameter.grad is None]

    # "unused" is the name, parameter of those don't see any gradient in backward().
    assert not unused, f"These parameters never received a gradient: {unused}"

  
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
    f"The shape of output tensor of _match_size should remain unchanged when the ups/encd spatial dimensions already match, " 
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

    assert torch.equal(matched_size, ups), "The output values of UNet._match_size should be identical to the input values when resizing is not needed."


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
    f"The height and width of output tensor of UNet._match_size should match the encoder tensor's ones, "
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
    f"The height and width of output tensor of UNet._match_size should match the encoder tensor's ones, "
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
    f"The height and width of output tensor of UNet._match_size should match the encoder tensor's ones, "
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
    f"The output tensor of UNet._match_size should preserve the data type of the input upsampled tensor, "
    f"Expected dtype {ups.dtype}, got {matched_size.dtype}"
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
    f"The output tensor of UNet._match_size should remain on the same device as the input upsampled tensor, "
    f"Expected device {ups.device}, got {matched_size.device}"
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
    f"Expected all output values of UNet._match_size remain constant value of 0.5, "
    f"but got minimum {out.min().item():.4f} and maximum {out.max().item():.4f}"
)


def test_match_size_matches_handy_calculated_bilinear_interpolation():
    """
    GIVEN: a 3x3 tensor upsampled to 4x4 (encoder tensor has the 4x4 spatial size)
    WHEN: _match_size is called
    THEN: the resized values match bilinear interpolation (align_corners=False)
          calculated by hand for each 16 output pixels
          
        How to calculate 4x4 tensor by hand
        -----------------------------------
        With align_corners=False, an output pixel index maps to the input
        coordinate:
    
            src = (output_index + 0.5) * (input_size / output_size) - 0.5
    
        - output_index values are (0, 1, 2, 3),

        - input_size/output_size = 3/4 = 0.75, converts output units to input units

        - that +0.5 moves to the pixel centre, 

        - that -0.5 shifts back to "index 0 = centre of input pixel 0". 
        
        - Bilinear interpolation then uses the two neighbouring input pixels of "src"
          weighted by how close "src" is to each. 
        
        - Coordinates outside the image are clamped to the border.

        - for a 3 -> 4 upsample this always works for the fixed pair
          of row/column weights, independently of height or width.

            output_index =0 : src = -0.125 -> clamped to 0 -> input index 0,         weight 1.0
            output_index =1 : src =  0.625 -> input index 0,1 with weights 0.375, 0.625
            output_index =2 : src =  1.375 -> input index 1,2 with weights 0.625, 0.375
            output_index =3 : src =  2.125 -> clamped to 2 -> input index 2,         weight 1.0
    
        bilinear interpolation is separable, so each output pixel is (row weight) x (column weight) applied to the index of 3x3 input.
        
        some calculation examples:

        (0,0): corner -> row0 w=1.0, col0 w=1.0          -> 1.0
    
        (0,1): row0 w=1.0, col blend 0.375*1 + 0.625*2   -> 1.625
    
        (1,1): 0.375*0.375*1 + 0.375*0.625*2
             + 0.625*0.375*4 + 0.625*0.625*5            -> 3.5
    
        (2,2): 0.625*0.625*5 + 0.625*0.375*6
             + 0.375*0.625*8 + 0.375*0.375*9            -> 6.5
    
        (3,3): corner -> row2 w=1.0, col2 w=1.0          -> 9.0
    """

    # A 2D matrix, reshaped with .view(1, 1, 3, 3) into the expected 4D (batch size=1, channels=1, height=3, width=3).
    ups = torch.tensor([[1., 2., 3.],
                        [4., 5., 6.],
                        [7., 8., 9.]]).view(1, 1, 3, 3)

    # create a tensor for encoder while spatial dimensions (h=4, w=4) matters here, not values.
    encd = torch.randn(1, 1, 4, 4)

    # _match_size should upsample ups from 3x3 to encd with 4x4 using bilinear interpolation with align_corners=False.
    matched_size = UNet._match_size(ups, encd)

    # A 2D matrix, reshaped with .view(1, 1, 4, 4) into the expected 4D (batch size=1, channels=1, height=4, width=4)
    # values are calculated by hand from the row/column weights described in the docstring. 
    handy_calculated = torch.tensor([[1.,     1.625, 2.375, 3.    ],
                                  [2.875,  3.5,   4.25,  4.875 ],
                                  [5.125,  5.75,  6.5,   7.125 ],
                                  [7.,     7.625, 8.375, 9.    ]]).view(1, 1, 4, 4)

    # Compare them with a tolerance for floating-point round-off errors of atol=1e-6.
    assert torch.allclose(matched_size, handy_calculated, atol=1e-6), (
    f"The UNet._match_size output doesn't match the handy-calculated values of the bilinear interpolation, "
    f"expected {handy_calculated}, got {matched_size}"
    )
