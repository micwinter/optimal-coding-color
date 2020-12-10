# import packages
import os
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image
import numpy as np

def sample_patches(image, num_patches, patch_size):
    """
    Sample N square MxMxD patches from an input image and return them in an NxDxMxM matrix.
    image (np.array): Image to sample from, of size LxWxD, (length, width, num_channels)
    num_patches (int): Number of patches to sample from the input image
    patch_size (int): Size of the patch
    returns: NxDxMxM matrix of samples from the image
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

    return np.stack(samples).transpose(0,3,1,2)

# utility functions
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def save_decoded_image(img, epoch, save_path):
    img = img.view(img.size(0), 1, 60, 60)
    save_image(img, os.path.join(save_path, 'linear_ae_image{}.png'.format(epoch)))


def viz_image_reconstruction(net, testloader):
     for batch in testloader:
        batch = batch.to(device)
        batch = batch.view(batch.size(0), -1)
        outputs = net(batch)
        outputs = outputs.view(outputs.size(0), 1, 60, 60).cpu().data
        save_image(outputs, 'autoencoder_reconstruction.png')
        break