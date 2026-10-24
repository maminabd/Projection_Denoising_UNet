
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):

    """
    projecting block (e.g. in 1st layer projects 2 channels --to--> 32 channels) followed by a residual block.

    Architecture:
        Input(in_ch = 2) ->  Conv3x3(in_ch = 2 -> out_ch e.g. 32) -> [saved as residual]
                         ->  BN + ReLU
                         ->  Conv3x3(out_ch->out_ch) + BN + ReLU
                         ->  Conv3x3(out_ch->out_ch) + BN
                         ->  Add residual -> ReLU -> Output(out_ch e.g. 32)

    The first convolution changes the number of channels, so the residual
    connection can be added right after it without an extra 1x1 convolution.

    Args:
        in_channels: Number of input channels of the layer
        out_channels: Number of output channels of the layer
        (e.g. for 2nd layer in_channels=32 and out_channels=64)
    """

    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Project to the number of channels of the layer
        self.proj_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Convolution within the layer
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)


    def forward(self, x):

        out = self.proj_conv(x)
        residual = out # save it as residual.
        out = self.relu(self.bn(out))

        # 2 conv layers
        out = self.conv(out)
        out = self.relu(self.bn(out))
        out = self.bn(self.conv(out))
        out = self.relu(out + residual) # Add the residual and final activation

        return out


class UNet(nn.Module):
    """
    U-Net denoiser with residual blocks, for 2D projections.

    Args:
        in_channels: number of input channels (2: noisy LE and HE).
        out_channels: number of output channels (2: denoised LE and HE).
    """

    def __init__(self, in_channels=2, out_channels=2):
        super(UNet, self).__init__()

        # Encoder: 2 --> 32 --> 64 --> 128 --> 256
        self.enc1 = ResBlock(in_channels, 32) # 2 --to--> 32
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResBlock(32, 64) # 32 --to--> 64
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResBlock(64, 128) # 64 --to--> 128
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ResBlock(128, 256) # 128 --to--> 256
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bridge = ResBlock(256, 512) # 256 --to--> 512

        # Decoder: 512 --> 256 --> 128 --> 64 --> 32
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # upsampling (x2 dimension) and halving channels depth
        self.dec4 = ResBlock(512, 256)  # input: concatenates(enc4, up4) = 512

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ResBlock(256, 128)  # input: concatenates(enc3, up3) = 256

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ResBlock(128, 64)  # input: concatenates(enc2, up2) = 128

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ResBlock(64, 32)  # input: concatenates (enc1, up1) = 64

        # Output layer: 32 --to--> 16 --finally to--> out_channels = 2
        self.pre_final_conv = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.pre_final_bn = nn.BatchNorm2d(16)
        self.pre_final_relu = nn.ReLU(inplace=True)
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    @staticmethod
    def _match_size(ups, encd):

        """
        Resize Height (H) and Width (W) of upsampled tensor to match the spatial size of
        encoder tensor of that layer in case of any size mismatch.

        Args:
            ups (torch.Tensor): the (B, C, H, W) tensor for resizing its H, W. (B: batch size, C: number of channels).

            encd (torch.Tensor): encoder tensor with spatial dimension (H, W) of that layer

        Returns:
                torch.Tensor: tensor with the same content as "ups" but with spatial size matching the "encd".
        """

        if ups.shape[2:] != encd.shape[2:]:
            ups = F.interpolate(ups, size=encd.shape[2:], mode='bilinear', align_corners=False)
        return ups

    def forward(self, x):

        """
        Args:
            x: noisy projections, shape (B, in_channels, H, W).

        Returns:
            denoised projections, shape (B, out_channels, H, W)
        """

        # Encoder: the features of each level are kept for the skip connections
        enc1 = self.enc1(x)                       # (channels = 32,  H,    W)
        enc2 = self.enc2(self.pool1(enc1))         # (channels = 64,  H/2,  W/2)
        enc3 = self.enc3(self.pool2(enc2))         # (channels = 128, H/4,  W/4)
        enc4 = self.enc4(self.pool3(enc3))         # (channels = 256, H/8,  W/8)

        # Bottleneck
        bridge = self.bridge(self.pool4(enc4))     # (512, H/16, W/16)

        # Decoder: upsample, resize, concatenate the skip connection, convolve
        dec4 = self._match_size(self.up4(bridge), enc4)
        cat4 = torch.cat([enc4, dec4], dim=1)
        dec4 = self.dec4(cat4)

        dec3 = self._match_size(self.up3(dec4), enc3)
        cat3 = torch.cat([enc3, dec3], dim=1)
        dec3 = self.dec3(cat3)

        dec2 = self._match_size(self.up2(dec3), enc2)
        cat2 = torch.cat([enc2, dec2], dim=1)
        dec2 = self.dec2(cat2)

        dec1 = self._match_size(self.up1(dec2), enc1)
        cat1 = torch.cat([enc1, dec1], dim=1)
        dec1 = self.dec1(cat1)

        # Output
        output = self.pre_final_conv(dec1)
        output = self.pre_final_bn(output)
        output = self.pre_final_relu(output)
        output = self.final_conv(output)

        return output
