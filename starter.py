"Load in data"

import imageio
import numpy as np
import matplotlib.pyplot as plt
import os


data_root = '../data/val_256'

images = os.path.listdir(data_root)

# preprocess
# subtract the mean of each large image and rescale the image to attain a variance of 1 for the pixels

# input noise is 8dB (sigma_nz = 0.4)

# 100 neurons

# We initialized filter weights and nonlinearity coefficients to random Gaussian values. Batch size was 100 patches, resampled after each update of the parameters. We trained the model for 100,000 iterations of gradient ascent with fixed step size
