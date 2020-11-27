"Load in data"

import imageio
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

def sample_patches(image, num_patches, patch_size):
    """
    Sample N square MxM patches from an input image and return them in an NxMxM matrix.
    image (numpy array): Image to sample from
    num_patches (int): Number of patches to sample from the input image
    patch_size (int): Size of the patch
    returns: NxMxM matrix of samples from the image
    """
    # Get indices of image that are viable for sampling (don't sample indices that would cause a sample too far to an edge)
    h,w,c = image.shape
    orig_indices = np.arange(h*w)
    orig_indices = orig_indices.reshape(h,w)
    margin = int(np.ceil(patch_size/2))
    valid_indices = orig_indices[margin:h-margin, margin:w-margin].ravel()
    np.random.shuffle(valid_indices)
    sampled = valid_indices[:num_patches]
    sampled_indices = [np.where(orig_indices == x) for x in sampled]
    # create samples
    samples = [image[int(x)-margin:int(x)+margin, int(y)-margin:int(y)+margin,:] for x,y in sampled_indices]

    return np.stack(samples)


data_root = '../../data/val_256'

images = os.listdir(data_root)

# preprocess
# subtract the mean of each large image and rescale the image to attain a variance of 1 for the pixels
ims = [imageio.imread(os.path.join(data_root, x)) for x in images]

# input noise is 8dB (sigma_nz = 0.4)

# 100 neurons
num_patches = 100
patch_size = 16
# sample = sample_patches(ims[0], num_patches, patch_size)
samples = [sample_patches(im, num_patches, patch_size) for im in ims]
samples = np.stack(samples)

# We initialized filter weights and nonlinearity coefficients to random Gaussian values. Batch size was 100 patches, resampled after each update of the parameters. We trained the model for 100,000 iterations of gradient ascent with fixed step size
