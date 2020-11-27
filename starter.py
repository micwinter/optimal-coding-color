"Load in data"

import imageio
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

def sample_patches(image, num_patches, patch_size, color=False):
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
    if color:
        samples = [image[int(x)-margin:int(x)+margin, int(y)-margin:int(y)+margin,:] for x,y in sampled_indices]
    else:
        samples = [image[int(x)-margin:int(x)+margin, int(y)-margin:int(y)+margin] for x,y in sampled_indices]

    return np.stack(samples)

#def neur_nonlin():


data_root = '../../data/val_256'

images = os.listdir(data_root)

# preprocess
# subtract the mean of each large image and rescale the image to attain a variance of 1 for the pixels
ims = [imageio.imread(os.path.join(data_root, x)) for x in images]

# input noise is 8dB (sigma_nz = 0.4)

num_patches = 100
patch_size = 16
# sample = sample_patches(ims[0], num_patches, patch_size)
samples = [sample_patches(im, num_patches, patch_size, color=False) for im in ims]
samples = np.stack(samples)

# train
iters = 100000
step_size = 0.01
batch_size = 100
num_neurons = 100

init_filters = np.random.rand(num_neurons, patch_size, patch_size)
# TODO: assert filters are unit norm

# training iterations
for iter in range(iters):
    im_noise =  np.random.normal(0,0.1,1) # image noise
    resp_noise = np.random.normal(0,0.1,1)  # response noise
    generator_signal = np.dot(W.T, curr_samples+im_noise)
    responses = neur_nonlin(generator_signal) + resp_noise
    # TODO: slope of the nonlinearity (g_i) gj = dfj/dyj
    num_kernels = 500
    # slope nonlinearity
    coeffs = np.random.randint()
    accum = 0
    mus = np.arange(100)
    # sigmas = spaces for smooth overlap of kernels
    for k in range(num_kernels):
        kernel = coef*np.exp(-1*((responses-mus)^2/()))

    # objective function
    # Gi is a diagonal matrix containing the local derivatives of the response functions gj (yj) at yj(xi)
    # G = #TODO

    # If input and output noises are assumed to be constant and Gaussian, with covariances C_nx and C_nr
    # C_nx = input noise
    # C_nr = output noise
    likelihood = G*W.T*im_noise*W*G+resp_noise# C_rx
    C_xr = np.linalg.inv(np.cov(curr_samples)) + W*G*(np.linalg.inv(likelihood))*G*W.T
    h_x_r = (1/2)*np.log(2*np.pi*np.exp(1)*np.linalg.det(C_i_xr))

# TODO: neural nonlinearity (f_i)






# generator signal (y_i)

# objective function

# prior estimation

# likelihood estimation

# gradient ascent
