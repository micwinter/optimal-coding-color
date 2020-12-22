# import packages
import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import datetime

class LinearAutoencoder(nn.Module):
    """
    Starter version of the model. The model is linear, and takes in HxWxD, down to number of neurons.
    """
    def __init__(self, patch_size, num_neurons):
        super(LinearAutoencoder, self).__init__()

        in_features = 30*patch_size*patch_size

        # encoder
        self.enc = nn.Linear(in_features=in_features, out_features=num_neurons)
        # decoder
        self.dec = nn.Linear(in_features=num_neurons, out_features=in_features)

    def forward(self, x):
        # encoder
        x = F.relu(self.enc(x))
        # decoder
        x = F.relu(self.dec(x))
        return x

class LinearAESpectrum(nn.Module):
    """
    Linear autoencoder that tries to learn 3 cone wavelength sensitivies. The model is linear, and takes in HxWxD, down to number of cones*number of wavelengths (in channels).
    """
    def __init__(self, in_channels, patch_size, num_cones=3):
        super(LinearAESpectrum, self).__init__()

        in_features = in_channels*patch_size*patch_size
        out_features = num_cones*in_channels

        # encoder
        self.enc = nn.Linear(in_features=in_features, out_features=out_features)
        # decoder
        self.dec = nn.Linear(in_features=out_features, out_features=in_features)

    def forward(self, x):
        # encoder
        x = F.relu(self.enc(x))
        # decoder
        x = F.relu(self.dec(x))
        return x


class LinearAESpectrum1(nn.Module):
    """
    Linear autoencoder that tries to learn 3 cone wavelength sensitivies. The model is linear, and takes in HxWxD, down to number of cones*number of wavelengths (in channels).
    """
    def __init__(self, in_channels, patch_size, num_cones=3):
        super(LinearAESpectrum1, self).__init__()

        in_features = in_channels*patch_size*patch_size
        # out_features = num_cones*in_channels

        # encoder
        self.enc1 = nn.Linear(in_features=in_features, out_features=28)
        self.enc2 = nn.Linear(in_features=28, out_features=3)
        # decoder
        self.dec1 = nn.Linear(in_features=3, out_features=28)
        self.dec2 = nn.Linear(in_features=28, out_features=in_features)

    def forward(self, x):
        # encoder
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        # decoder
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        return x


class LinearAESpectrum2(nn.Module):
    """
    **Noise Added**
    Linear autoencoder that tries to learn 3 cone wavelength sensitivies. The model is linear, and takes in HxWxD, down to number of cones*number of wavelengths (in channels).
    """
    def __init__(self, in_channels, patch_size, num_cones=3, in_noise=1, out_noise=1):
        super(LinearAESpectrum2, self).__init__()

        in_features = in_channels*patch_size*patch_size
        self.in_noise = torch.normal(0, in_noise, size=(1, in_features)).cuda()
        self.out_noise = torch.normal(0, out_noise, size=(1, num_cones)).cuda()
        # out_features = num_cones*in_channels

        # encoder
        self.enc1 = nn.Linear(in_features=in_features, out_features=in_channels)
        self.enc2 = nn.Linear(in_features=in_channels, out_features=num_cones)
        # decoder
        self.dec1 = nn.Linear(in_features=num_cones, out_features=in_channels)
        self.dec2 = nn.Linear(in_features=in_channels, out_features=in_features)

    def forward(self, x):
        # encoder
        # import ipdb; ipdb.set_trace()
        x = x + self.in_noise
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        # decoder
        x = x + self.out_noise
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        return x


class ConvAutoencoder(nn.Module):
    """
    Second version of the model. The model is convolutional, and takes in HxWxD, down to number of neurons.
    """
    def __init__(self, in_channels=30, latent_size=3):
        super(ConvAutoencoder, self).__init__()
        # encoder
        # conv layer (depth from 30 --> 1), 3x3 kernels
        self.enc = nn.Conv2d(in_channels, latent_size, 3)
        # decoder
        self.dec = nn.ConvTranspose2d(latent_size, in_channels, 3)

    def forward(self, x):
        # encoder
        x = F.relu(self.enc(x))
        # decoder
        x = F.relu(self.dec(x)) # try with sigmoid or relu
        return x

def train_autoencoder(net, trainloader, testloader, patch_size, num_epochs, learning_rate, optimizer, device, criterion, savename):
    now = datetime.datetime.now() # current timestamp
    train_loss = []
    test_loss = []
    hold_tloss = np.inf

    for epoch_idx, epoch in enumerate(range(num_epochs)):
        running_loss = 0.0
        for batch in trainloader:
            batch = batch.to(device)
            batch = batch.view(batch.size(0), -1) # if linear autoencoder
            # if conv autoencoder
            optimizer.zero_grad()
            outputs = net(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


        loss = running_loss / len(trainloader)
        train_loss.append(loss)

        # Test loss
        # with torch.no_grad:
        running_test_loss = 0.0
        for batch in testloader:
            batch = batch.to(device)
            batch = batch.view(batch.size(0), -1) # if linear autoencoder
            # if conv autoencoder
            # optimizer.zero_grad()
            outputs = net(batch)
            curr_test_loss = criterion(outputs, batch)
            running_test_loss += curr_test_loss.item()

        tloss = running_test_loss / len(testloader)
        test_loss.append(tloss)

        # Save model state if test loss is lowest
        if tloss < hold_tloss:
            hold_tloss = tloss
            torch.save({
              'state_dict': net.state_dict(),
              'optimizer': optimizer.state_dict(),
              'loss': train_loss
              }, savename)

        print('    Epoch {} of {}, Train Loss: {:.3f}, Test Loss: {:.3f}'.format(
            epoch+1, num_epochs, train_loss[epoch_idx], test_loss[epoch_idx]))
        #if epoch % 5 == 0:
        #    save_decoded_image(outputs.cpu().data, epoch, IM_PATH)
    return train_loss, test_loss
