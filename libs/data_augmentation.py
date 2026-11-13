
import torch
import random

def rotation_augmentation(input, target, k):
        """
        Apply rotation augmentation using torch.rot90
        Rotation angles: 90°, 180°, 270°, 360°

        Args:
            input_tensor: Input tensor (C=2, H=300, W=300) - LE and HE channels 
            output_tensor: Output tensor (C=2, H=300, W=300) - LE and HE channels 

        Returns:
            Rotated tensors
        """

        # Randomly choose rotation: 1 (90°), 2 (180°), 3 (270°), 4 (360°)
        rotation_angles = [1, 2, 3, 4]  
        k = random.choice(rotation_angles)

        # torch.rot90 rotates counter-clockwise in the plane specified by dims (H, W axes = [1, 2])
        input = torch.rot90(input, k, dims=[1, 2])
        target = torch.rot90(target, k, dims=[1, 2])

        return input, target
