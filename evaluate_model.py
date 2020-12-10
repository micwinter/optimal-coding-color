# import packages
import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.utils import save_image

from spectral import *
import numpy as np
import datetime

from encoders import LinearAutoencoder, ConvAutoencoder, LinearAESpectrum, train_autoencoder

PATCH_SIZE=16

# net = LinearAutoencoder()
net = LinearAESpectrum(in_channels=28, patch_size=PATCH_SIZE)


checkpoint = torch.load('checkpoint_grid_ps_16_lr_0.0001_linaespec.t7')
net.load_state_dict(checkpoint['state_dict'])
# import ipdb; ipdb.set_trace()

IM_PATH = '/media/big_hdd/opt-color/landscapes_fla'  # YOUR DATA PATH HERE
train_im = envi.open(os.path.join(IM_PATH, 'hyperspectral_1.hdr'), os.path.join(IM_PATH, 'hyperspectral_1.fla'))

shared_wavelength = np.arange(430, 730, 10)

# import ipdb; ipdb.set_trace()
temp = torch.Tensor(train_im[:16,:16,:28])
temp = temp[np.newaxis,:]
# out = net.enc(temp.permute(0,3,1,2))
# out_im = out.detach().numpy()[0,:,:,:].transpose(1,2,0)
# out_im = out_im/np.max(out_im)
# plt.imshow(out_im)
# plt.savefig('output.png')

# For linear AE sprectrum Evaluation
out = net.enc(temp.reshape(1,-1))

out = out.detach().numpy()
for ii in range(3):
    plt.plot(out[ii*28:(ii+1)*28])
import ipdb; ipdb.set_trace()
plt.legend(['Cone 1', 'Cone 2', 'Cone 3'])
plt.savefig('output.png')
