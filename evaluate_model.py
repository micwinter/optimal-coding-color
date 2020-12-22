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

ext = 'linaespec2_noise'

PATCH_SIZE = 16
LEARNING_RATE = 1e-3

now = '2020-12-21 22:56:39.334282'

net = LinearAESpectrum2(
        in_channels=31,
        patch_size=PATCH_SIZE,
        num_cones=NUM_CONES,
        in_noise=IN_NOISE_SIGMA,
        out_noise=OUT_NOISE_SIGMA
        )

savename = f'{ext}_{now}_checkpoint_grid_ps_{PATCH_SIZE}_lr_{LEARNING_RATE}.t7'
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
plt.savefig(f'{ext}_{now}_weights.png')






# -------------------

# net = LinearAutoencoder()
# net = ConvAutoencoder()
# # net = LinearAESpectrum1(in_channels=28, patch_size=PATCH_SIZE)
#
#
# # checkpoint = torch.load('checkpoint_grid_ps_16_lr_0.0001_linaespec.t7')
# # checkpoint = torch.load('checkpoint_grid_ps_16_lr_0.0001_linaespec1.t7')
# checkpoint = torch.load('checkpoint_grid_ps_60_lr_0.0001.t7')
# net.load_state_dict(checkpoint['state_dict'])
# # import ipdb; ipdb.set_trace()
#
# IM_PATH = '/media/big_hdd/opt-color/landscapes_fla'  # YOUR DATA PATH HERE
# train_im = envi.open(os.path.join(IM_PATH, 'hyperspectral_1.hdr'), os.path.join(IM_PATH, 'hyperspectral_1.fla'))
#
# # shared_wavelength = np.arange(430, 730, 10)
#
# # import ipdb; ipdb.set_trace()
# temp = torch.Tensor(train_im[:,:,:])
# temp = temp[np.newaxis,:]
# out = net.enc(temp.permute(0,3,1,2))
# out_im = out.detach().numpy()[0,:,:,:].transpose(1,2,0)
# out_im = out_im/np.max(out_im)
# plt.imshow(np.stack((out_im[:,:,1], out_im[:,:,2], out_im[:,:,0])).transpose(1,2,0))
# plt.savefig('output_conv6.png')


# temp = torch.Tensor(train_im[:16,:16,:28])
# temp = temp[np.newaxis,:]
# # For linear AE sprectrum Evaluation
# out = F.relu(net.enc1(temp.reshape(1,-1)))
#
# out = out.detach().numpy()
# np.save('out.npy', out)
# for ii in range(3):
#     plt.plot(out[0,ii*28:(ii+1)*28].T)
# # import ipdb; ipdb.set_trace()
# locs, labels = plt.xticks()
# plt.legend(['Cone 1', 'Cone 2', 'Cone 3'])
# plt.xticks(np.arange(28), np.arange(430, 710, 10))
# plt.xlabel('Wavelength')
# plt.savefig('output.png')

# # For linear AE sprectrum 1 Evaluation
# # import ipdb; ipdb.set_trace()
# weights = net.enc2.weight.data.detach().numpy()
# fig = plt.figure(figsize=(15,5))
# plt.plot(weights.T)
# # out = F.relu(net.enc1(temp.reshape(1,-1)))
# #
# # out = out.detach().numpy()
# # np.save('out.npy', out)
# # for ii in range(3):
# #     plt.plot(out[0,ii*28:(ii+1)*28].T)
# # import ipdb; ipdb.set_trace()
# # locs, labels = plt.xticks()
# plt.legend(['Cone 1', 'Cone 2', 'Cone 3'])
# plt.xticks(np.arange(28), np.arange(430, 710, 10))
# plt.xlabel('Wavelength')
# plt.savefig('output_linspec1_weights.png')
