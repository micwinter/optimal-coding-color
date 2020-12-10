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

from utilities import sample_patches, get_device, save_decoded_image, viz_image_reconstruction
from encoders import LinearAutoencoder, ConvAutoencoder, train_audoencoder

# constants for data loading
BATCH_SIZE = 128
IM_PATH = 'data/landscapes_fla'
NUM_PATCHES = 1000
PATCH_SIZE = 60

# Load image as array
train_im = envi.open(os.path.join(IM_PATH, 'landscape01.hdr'), os.path.join(IM_PATH, 'landscape01.fla'))
test_im = envi.open(os.path.join(IM_PATH,'landscape02.hdr'), os.path.join(IM_PATH,'landscape02.fla'))
# Get samples
trainset = sample_patches(train_im, NUM_PATCHES, PATCH_SIZE)
testset = sample_patches(test_im, NUM_PATCHES, PATCH_SIZE)

trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
testloader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# get the computation device
device = get_device()
print(f'device: {device}')

# constants
NUM_EPOCHS = 5
grid_LEARNING_RATE = [1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4]

# train the network
for LEARNING_RATE in grid_LEARNING_RATE:
    print(f'Learning rate: {LEARNING_RATE}')
    net = LinearAutoencoder()
    #net = ConvAutoencoder()
    #print(net)
    criterion = nn.MSELoss() 
    # load the neural network onto the device
    net.to(device)
    train_loss = train_audoencoder(net, trainloader, NUM_EPOCHS, LEARNING_RATE, 
        device=device, criterion=criterion)



#plt.figure()
#plt.plot(train_loss)
#plt.title('Train Loss')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.savefig('autoencoder_loss.png')
# test the network
# viz_image_reconstruction(net, testloader)
