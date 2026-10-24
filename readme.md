# Projection_Denoising_UNet
This repository contains a 2D U-Net for denoising dual-energy X-ray projections (Low Energy and High Energy channels), written in PyTorch.

## Overview
The model is a 2D U-Net with residual blocks. It takes as input noisy projections with 2 channels (LE and HE) of size 300 x 300 and returns as output the denoised projection with the same shape.

### Prepare Data
The model expects two directories of `.npy` files: one with the noisy inputs and one with the clean targets. Each input file must correspound with a target file with the same name.

Each file is a CBCT projection of a mouse with shape `(300, 300, 2)`. The last axis is the two channels: index 0 is LE (Low Energy projection) and index 1 is HE (High Energy projection). All projections are expected to already be normalized to the `[0, 1]` range.
