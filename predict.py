
import argparse
import glob
import os
import numpy as np
import torch
from libs.dataset import load_as_tensor
from libs.unet import UNet
from libs.utils import get_device

IMAGE_SHAPE = (300, 300, 2)  # (H, W, C) for each single projection (C: channels, LE and HE projections)

def parse_args():
    """Command-line arguments: path to noisy projections, trained denoiser model, and output for denoised projections."""
    parser = argparse.ArgumentParser(description="Denoise unseen projections with a trained U-Net")
    parser.add_argument('--noisy', required=True,
                        help="Directory with the noisy .npy files to denoise them")
    parser.add_argument('--model', required=True,
                        help="Trained model (.pth) saved by training.py")
    parser.add_argument('--output', default='test_results',
                        help="Directory where the results are saved")
    return parser.parse_args()


def main():

    args = parse_args()

    device = get_device()

    # gather all noisy .npy files of the given directory and sort them to be processed
    noisy_dataset = sorted(glob.glob(os.path.join(args.noisy,"*.npy")))
    
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model '{args.model}' not found")

    # loads a checkpoint
    # weights_only=True: the checkpoint only holds tensors and numbers
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)

    # instantiate the UNet model and bring it to GPU
    model = UNet(in_channels=2, out_channels=2).to(device)

    # load the trained weights from the checkpoint into the Unet model
    model.load_state_dict(checkpoint['model_state_dict'])

    # switch the model to evaluation mode, inference only
    model.eval()
    
    os.makedirs(args.output, exist_ok=True)

    for noisy_projection in noisy_dataset:
        
        # load the noisy npy as a tensor of shape (C, H, W)
        noisy_tensor = load_as_tensor(noisy_projection) 

        # unet model was built to expect (N=1, C, H, W), Hence add the batch axis at index 0 to make the tensor
        batch_noisy_tensor = noisy_tensor.unsqueeze(0)

        # bring it to GPU
        batch_noisy_tensor= batch_noisy_tensor.to(device)

        # disable gradient calculations, we're only predicting, no backpropagation needed
        with torch.no_grad():
            output = model(batch_noisy_tensor) # forward to UNet: model returns a GPU tensor of shape (1, C, H, W)

        # remove the batch dimension and make it (C, H, W), and ...
        # bring back the tensor to CPU for plotting, saving as .npy, etc
        denoised = output.squeeze(0).cpu()
        
        # save the denoised tensor as a .npy file
        np.save (os.path.join(args.output, os.path.basename(noisy_projection)), denoised.numpy())
        
    print(f"All noisy projections in {args.noisy} have been denoised and saved in {args.output}")

if __name__ == '__main__':
    main()
