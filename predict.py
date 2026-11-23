
import argparse
import os
import numpy as np
import torch
from libs.dataset import check_files, find_paired_files, load_as_tensor
from libs.unet import UNet
from libs.utils import (get_device, projs_comparison, absolute_difference,
                        mean_absolute_difference)

def parse_args():
    """Command-line arguments: path to noisy projections, trained denoiser model, and output for denoised projections."""
    parser = argparse.ArgumentParser(description="Denoise unseen projections with a trained U-Net")
    parser.add_argument('--noisy', required=True,
                        help="Directory with the noisy .npy files to denoise them")
    parser.add_argument('--groundtruth', required=True,
                        help="Directory with the noiseless ground truth .npy files (same names as --noisy)")
    parser.add_argument('--model', required=True,
                        help="Trained model (.pth) saved by training.py")
    parser.add_argument('--output', default='./predicted_results',
                        help="Directory where the results will be saved (default: ./predicted_results)")
    return parser.parse_args()


def main():

    args = parse_args()

    device = get_device()

    noisy_files, groundtruth_files = find_paired_files(args.noisy, args.groundtruth)
    check_files(noisy_files, groundtruth_files)

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
    all_differences = []
    for noisy_projection, groundtruth_projection in zip(noisy_files, groundtruth_files):
        
        # load the noisy npy as a tensor of shape (C, H, W)
        noisy_tensor = load_as_tensor(noisy_projection) 

        # unet model was built to expect (N=1, C, H, W), 
        # Hence add the batch axis at index 0 to make the tensor and then bring it to GPU
        batch_noisy_tensor = noisy_tensor.unsqueeze(0).to(device)

        # disable gradient calculations, we're only predicting, no backpropagation needed
        with torch.no_grad():
            output = model(batch_noisy_tensor) # forward to UNet: model returns a GPU tensor of shape (1, C, H, W)

        # remove the batch dimension and make it (C, H, W), and ...
        # bring back the tensor to CPU for plotting, saving as .npy, etc
        denoised = output.squeeze(0).cpu()
        
        # save the denoised tensor as a .npy file
        np.save (os.path.join(args.output, os.path.basename(noisy_projection)), denoised.numpy())

        ground_truth = load_as_tensor(groundtruth_projection)

        full_name = os.path.basename(noisy_projection) # it gives the filename with its extension, e.g. "proj_1.npy"
        full_name_split = os.path.splitext(full_name) # it split the filename and gives a tuple: ("proj_1", ".npy")
        name = full_name_split[0] # just the main name without the extension, e.g. "proj_1" for name of comparison picture
        comparison_pic_name = os.path.join(args.output, f"{name}.png")

        # absolute difference maps and their means over the pixels of each channel, [mean_LE, mean_HE]
        difference = absolute_difference(denoised, ground_truth)
        mean_differences = mean_absolute_difference(denoised, ground_truth)

        projs_comparison(noisy_tensor, denoised, ground_truth, difference, mean_differences,
                         filename=comparison_pic_name, title=name)

        all_differences.append(mean_differences)

        print(f"{name}: mean absolute difference: LE {mean_differences[0]:.6f}, HE {mean_differences[1]:.6f}")

    average = [sum(values) / len(all_differences) for values in zip(*all_differences)]

    print(f"\nAverage over {len(noisy_files)} evaluation files: LE {average[0]:.6f}, HE {average[1]:.6f}")
    
    print(f"\nAll noisy projections in {args.noisy} have been denoised and saved in {args.output}")

if __name__ == '__main__':
    main()
