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
from encoders import LinearAutoencoder, ConvAutoencoder, LinearAESpectrum, LinearAESpectrum1, train_autoencoder
# from data_loading import data_initializer

# constants for data loading
IM_PATH = '/media/big_hdd/opt-color/landscapes_fla'  # YOUR DATA PATH HERE
BATCH_SIZE = 128
NUM_PATCHES = 1000
grid_PATCH_SIZE = [16]#[16, 60, 100]
# constants for autoencoder
NUM_EPOCHS = 1000
NUM_NEURONS = 100
grid_LEARNING_RATE = [1e-4]#[1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

ext = 'linaespec1'

for PATCH_SIZE in grid_PATCH_SIZE:
    for LEARNING_RATE in grid_LEARNING_RATE:

        print(f'Patch size: {PATCH_SIZE}, Learning rate: {LEARNING_RATE}')

        # trainloader, testloader = data_initializer(patch_size=PATCH_SIZE)
        trainloader, num_channels = data_initializer(patch_size=PATCH_SIZE)
        # get the computation device
        device = get_device()
        print(f'device: {device}')

        # train the network
        # net = LinearAutoencoder(patch_size=PATCH_SIZE, num_neurons=NUM_NEURONS)
        # net = ConvAutoencoder(in_channels=num_channels)
        net = LinearAESpectrum1(in_channels=num_channels, patch_size=PATCH_SIZE)
        #print(net)
        criterion = nn.MSELoss()
        # load the neural network onto the device
        net.to(device)
        train_loss = train_autoencoder(net, trainloader, PATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
            device=device, criterion=criterion)
        plt.figure()
        plt.plot(train_loss)
        plt.title('Train Loss | PS {} | LR {}'.format(PATCH_SIZE, LEARNING_RATE))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig('autoencoder_loss_ps_{}_lr_{}_{}.png'.format(PATCH_SIZE, LEARNING_RATE, ext))
        plt.close()
        now = datetime.datetime.now() # current timestamp
        torch.save({
          'state_dict': net.state_dict(),
#           'optimizer': optimizer.state_dict(),
          'loss': train_loss
          }, 'checkpoint_grid_ps_{}_lr_{}_{}.t7'.format(PATCH_SIZE, LEARNING_RATE, ext))
