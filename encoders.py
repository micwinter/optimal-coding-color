# import packages
import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F

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

def train_autoencoder(net, trainloader, patch_size, num_epochs, learning_rate, device, criterion):
    train_loss = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in trainloader:
            batch = batch.to(device)
            batch = batch.view(batch.size(0), -1) # if linear autoencoder
            # if conv autoencoder
            optimizer = optim.Adam(net.parameters(), lr=learning_rate)
            optimizer.zero_grad()
            outputs = net(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        print('    Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch+1, num_epochs, loss))
        #if epoch % 5 == 0:
        #    save_decoded_image(outputs.cpu().data, epoch, IM_PATH)
    return train_loss
