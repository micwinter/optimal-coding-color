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


class ConvAutoencoder(nn.Module):
    """
    Second version of the model. The model is convolutional, and takes in HxWxD, down to number of neurons.
    """
    def __init__(self, in_features=30, latent_size=3):
        super(ConvAutoencoder, self).__init__()
        # encoder
        # conv layer (depth from 30 --> 1), 3x3 kernels
        self.enc = nn.Conv2d(in_features, latent_size, 3)
        # decoder
        self.dec = nn.ConvTranspose2d(latent_size, in_features, 3)

    def forward(self, x):
        # encoder
        x = F.relu(self.enc(x))
        # decoder
        x = F.relu(self.dec(x))
        # x = torch.sigmoid(self.dec(x)) # try with sigmoid or relu
        return x

# net = LinearAutoencoder()
net = ConvAutoencoder()


checkpoint = torch.load('checkpoint_2020-12-09 18:51:42.135329.t7')
net.load_state_dict(checkpoint['state_dict'])
# import ipdb; ipdb.set_trace()

IM_PATH = '/media/big_hdd/opt-color/landscapes_fla'  # YOUR DATA PATH HERE
train_im = envi.open(os.path.join(IM_PATH, 'landscape01.hdr'), os.path.join(IM_PATH, 'landscape01.fla'))

temp = torch.Tensor(train_im[:,:,:])
temp = temp[np.newaxis,:]
out = net.enc(temp.permute(0,3,1,2))
out_im = out.detach().numpy()[0,:,:,:].transpose(1,2,0)
out_im = out_im/np.max(out_im)
plt.imshow(out_im)
plt.savefig('output.png')
