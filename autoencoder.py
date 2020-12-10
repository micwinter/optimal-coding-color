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

from utilities import sample_patches, get_device, save_decoded_image, viz_image_reconstruction
from encoders import LinearAutoencoder, ConvAutoencoder, train_audoencoder
from data_loading import data_initializer

def save_decoded_image(img, epoch, save_path):
    img = img.view(img.size(0), 1, 28, 28)
    save_image(img, os.path.join(save_path, 'linear_ae_image{}.png'.format(epoch)))

# constants for data loading
IM_PATH = '/media/big_hdd/opt-color/landscapes_fla'  # YOUR DATA PATH HERE
BATCH_SIZE = 128
NUM_PATCHES = 1000
grid_PATCH_SIZE = [16, 60, 100]
# constants for autoencoder
NUM_EPOCHS = 10000
NUM_NEURONS = 100
grid_LEARNING_RATE = [1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

for PATCH_SIZE in grid_PATCH_SIZE:
    for LEARNING_RATE in grid_LEARNING_RATE:
    
        print(f'Patch size: {PATCH_SIZE}, Learning rate: {LEARNING_RATE}')

        trainloader, testloader = data_initializer(patch_size=PATCH_SIZE)  
        # get the computation device
        device = get_device()
        print(f'device: {device}')
    
        # train the network
        net = LinearAutoencoder(patch_size=PATCH_SIZE, num_neurons=NUM_NEURONS)
        #net = ConvAutoencoder()
        #print(net)
        criterion = nn.MSELoss() 
        # load the neural network onto the device
        net.to(device)
        train_loss = train_audoencoder(net, trainloader, PATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, 
            device=device, criterion=criterion)
        now = datetime.datetime.now() # current timestamp
        torch.save({
          'state_dict': net.state_dict(),
#           'optimizer': optimizer.state_dict(),
          'loss': train_loss
          }, 'checkpoint_{}.t7'.format(now))

