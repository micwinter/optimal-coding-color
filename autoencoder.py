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

from spectral import *
import numpy as np
import datetime

from utilities import sample_patches, get_device, save_decoded_image, viz_image_reconstruction, data_initializer
from encoders import LinearAutoencoder, ConvAutoencoder, LinearAESpectrum, LinearAESpectrum1, train_autoencoder, LinearAESpectrum2
# from data_loading import data_initializer

# constants for data loading
IM_PATH = '/media/big_hdd/opt-color/hyperspec_ims'  # YOUR DATA PATH HERE
BATCH_SIZE = 128
NUM_PATCHES = 100
# grid_PATCH_SIZE = [16]#[16, 60, 100]
# constants for autoencoder
NUM_EPOCHS = 10000
# NUM_NEURONS = 100
NUM_CONES = 3
IN_NOISE_SIGMA = 0.4
OUT_NOISE_SIGMA = 2
# grid_LEARNING_RATE = [1e-4]#[1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

ext = 'linaespec2_noise'

PATCH_SIZE = 16
LEARNING_RATE = 1e-3

now = datetime.datetime.now() # current timestamp
savename = f'{ext}_{now}_checkpoint_grid_ps_{PATCH_SIZE}_lr_{LEARNING_RATE}.t7'


# for PATCH_SIZE in grid_PATCH_SIZE:
#     for LEARNING_RATE in grid_LEARNING_RATE:

print(f'Patch size: {PATCH_SIZE}, Learning rate: {LEARNING_RATE}')

# trainloader, testloader = data_initializer(patch_size=PATCH_SIZE)
trainloader, testloader, num_channels = data_initializer(patch_size=PATCH_SIZE, num_patches=NUM_PATCHES)
# get the computation device
device = get_device()
print(f'device: {device}')

# train the network
# net = LinearAutoencoder(patch_size=PATCH_SIZE, num_neurons=NUM_NEURONS)
# net = ConvAutoencoder(in_channels=num_channels)
net = LinearAESpectrum2(
        in_channels=num_channels,
        patch_size=PATCH_SIZE,
        num_cones=NUM_CONES,
        in_noise=IN_NOISE_SIGMA,
        out_noise=OUT_NOISE_SIGMA
        )
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
# load the neural network onto the device
net.to(device)
train_loss, test_loss = train_autoencoder(
                net,
                trainloader,
                testloader,
                PATCH_SIZE,
                NUM_EPOCHS,
                LEARNING_RATE,
                optimizer,
                device,
                criterion,
                savename)

# Plot of train loss
plt.figure()
plt.plot(train_loss)
plt.plot(test_loss)
plt.legend(['Train Loss', 'Test Loss'])
plt.title(f'Train/Test Loss | PS {PATCH_SIZE} | LR {LEARNING_RATE}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(f'{ext}_{now}_autoencoder_loss_ps_{PATCH_SIZE}_lr_{LEARNING_RATE}.png')
plt.close()
