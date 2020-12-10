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

class ConvAutoencoder(nn.Module):
    """
    Second version of the model. The model is convolutional, and takes in HxWxD, down to number of neurons.
    """
    def __init__(self, in_features=30, num_neurons=100):
        super(ConvAutoencoder, self).__init__()
        # encoder
        # conv layer (depth from 30 --> 1), 3x3 kernels
        self.enc = nn.Conv2d(30, 1, 3)
        # decoder
        self.dec = nn.ConvTranspose2d(1, 30, 3)

    def forward(self, x):
        # encoder
        x = F.relu(self.enc(x))
        # decoder
        x = torch.sigmoid(self.dec(x)) # try with sigmoid or relu
        return x

def train_audoencoder(net, trainloader, patch_size, num_epochs, learning_rate, device, criterion):
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