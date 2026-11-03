# Projection_Denoising_UNet
This repository contains a 2D U-Net for denoising dual-energy X-ray projections (Low Energy and High Energy channels), written in PyTorch.

## Overview
The model is a 2D U-Net with residual blocks. It takes as input noisy projections with 2 channels (LE and HE) of size 300 x 300 and returns as output the denoised projection with the same shape.

### Prepare Data
The model expects two directories of `.npy` files: one with the noisy inputs and one with the clean targets. Each input file must correspound with a target file with the same name.

Each file is a CBCT projection of a mouse with shape `(300, 300, 2)`. The last axis is the two channels: index 0 is LE (Low Energy projection) and index 1 is HE (High Energy projection). All projections are expected to already be normalized to the `[0, 1]` range.

### Training
Once the data is located, train the model with the training script. The input and target directories are required:

```bash
python training.py --input "path/to/input" --target "path/to/target"
```

You can also set the following training parameters:
- Epochs (--epochs): default value 100
- Batch size (--batch): default value 8
- Save a checkpoint every N epochs (--save_every): default value 25


Run `python training.py --help` to see the full list. The remaining settings (learning rate, scheduler parameters and validation fraction) are constants at the top of [training.py](training.py).

## Contents
Projection Denoising U-Net is composed of scripts and modules:
- Script trains the model
- Modules define the dataset, the model, and the helper functions in PyTorch.

**Script:** 

| **Scripts** | **Description** |
|:----------:|:----------------|
| [training](./training.py) | Trains the model with the given input and target projections, epochs and batch size, learning rate regime and some other hyperparameters.<br>• Saves the best model based on improvement of validation loss<br>• Makes a log file of loss values during training<br>• Saves a plot of the loss history and learning rate per epoch |

**Modules:**

| **Module** | **Description** |
|:----------|:---------------|
| [dataset](libs/dataset.py) | Checks the data files and loads input/target projection pairs, train/validation split, create a batch of projections|
| [unet](libs/unet.py) | The model: residual block and the 2D U-Net built from them |
| [utils](libs/utils.py) | Helper functions: device setup, save/load checkpoints and history plots |


### Installation & Setup

To install the software, clone this repository:

```bash
git clone https://github.com/maminabd/Projection_Denoising_UNet.git
```

To be able to make the model run is mandatory to install the requirements via [requirements.txt](https://github.com/maminabd/Projection_Denoising_UNet/blob/main/requirements.txt).
We recommend installing libraries in a Conda environment. After activating the environment, run:

```bash
cd Projection_Denoising_UNet
conda install --file requirements.txt
```

### Testing
The group of tests are in the [tests](./tests/) directory including (`test_dataset.py`, `test_unet.py`). They are written with [pytest](https://docs.pytest.org/), which already installed via requirements.txt, and do not require a GPU or the real dataset, the files they need are generated in a temporary directory.

To run the whole group of tests, from the repository root:

```bash
cd Projection_Denoising_UNet
pytest ./tests/
```

To run a single test file or a single test:

```bash
pytest tests/test_dataset.py
pytest tests/test_dataset.py::test_split_files_sizes
```
