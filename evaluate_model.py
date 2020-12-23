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

from encoders import LinearAutoencoder, ConvAutoencoder, LinearAESpectrum,LinearAESpectrum1, LinearAESpectrum2, train_autoencoder

BATCH_SIZE = 128
NUM_PATCHES = 100

NUM_EPOCHS = 1000
NUM_CONES = 3
IN_NOISE_SIGMA = 0.4
OUT_NOISE_SIGMA = 2

# ext = 'linaespec2_noise'
# ext = 'linaespec2_noise_innoise_0.5_outnoise_0.5'
ext = 'linaespec2_noise_innoise_0.1_outnoise_0.01'

PATCH_SIZE = 16
LEARNING_RATE = 1e-3

# now = '2020-12-21 22:56:39.334282'
# now = '2020-12-21 23:14:00.882295'

net = LinearAESpectrum2(
        in_channels=31,
        patch_size=PATCH_SIZE,
        num_cones=NUM_CONES,
        in_noise=IN_NOISE_SIGMA,
        out_noise=OUT_NOISE_SIGMA
        )

# savename = f'{ext}_{now}_checkpoint_grid_ps_{PATCH_SIZE}_lr_{LEARNING_RATE}.t7'
savename = f'{ext}_checkpoint_grid_ps_{PATCH_SIZE}_lr_{LEARNING_RATE}.t7'
checkpoint = torch.load(savename)
net.load_state_dict(checkpoint['state_dict'])

# Plot of encoder weights
weights = net.enc2.weight.data.cpu().detach().numpy()
fig = plt.figure(figsize=(15,5))
plt.plot(weights.T)
locs, labels = plt.xticks()
plt.legend(['Cone 1', 'Cone 2', 'Cone 3'])
plt.xticks(np.arange(31), np.arange(400, 710, 10))
plt.ylabel('Learned Weights')
plt.xlabel('Wavelength')
plt.title('Learned Weights of Autoencoder Latent Space')
# plt.savefig(f'{ext}_{now}_weights.png')
plt.savefig(f'{ext}_weights.png')
