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



def sample_patches(image, num_patches, patch_size):
    """
    Sample N square MxMxD patches from an input image and return them in an NxDxMxM matrix.
    image (np.array): Image to sample from, of size LxWxD, (length, width, num_channels)
    num_patches (int): Number of patches to sample from the input image
    patch_size (int): Size of the patch
    returns: NxDxMxM matrix of samples from the image
    """
    # Get indices of image that are viable for sampling (don't sample indices that would cause a sample too far to an edge)
    h,w,c = image.shape
    orig_indices = np.arange(h*w)
    orig_indices = orig_indices.reshape(h,w)
    margin = int(np.ceil(patch_size/2))
    valid_indices = orig_indices[margin:h-margin, margin:w-margin].ravel()
    np.random.shuffle(valid_indices)
    sampled = valid_indices[:num_patches]
    sampled_indices = [np.where(orig_indices == x) for x in sampled]
    # create samples
    samples = [image[int(x)-margin:int(x)+margin, int(y)-margin:int(y)+margin,:] for x,y in sampled_indices]

    return np.stack(samples).transpose(0,3,1,2)

def save_decoded_image(img, epoch, save_path):
    img = img.view(img.size(0), 1, 28, 28)
    save_image(img, os.path.join(save_path, 'linear_ae_image{}.png'.format(epoch)))

# constants
NUM_EPOCHS = 10000
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
IM_PATH = '/media/big_hdd/opt-color/landscapes_fla'  # YOUR DATA PATH HERE
NUM_PATCHES = 1000
PATCH_SIZE = 16
# # image transformations
# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])

def load_flas(root=IM_PATH, list_of_files):
    files = []
    for file_ in list_of_files:
        files.append(
        envi.open(os.path.join(IM_PATH, '{}.hdr'.format(file_)),
        os.path.join(IM_PATH, '{}.fla'.format(file_))))

#TODO: Create batches by randomly subsampling image
# Load image as array
# train_im = envi.open(os.path.join(IM_PATH, 'landscape01.hdr'), os.path.join(IM_PATH, 'landscape01.fla'))
# test_im = envi.open(os.path.join(IM_PATH,'landscape02.hdr'), os.path.join(IM_PATH,'landscape02.fla'))
flas = load_flas(['landscape01', 'landscape02'])
# Get samples
# trainset = np.concatenate((sample_patches(train_im, NUM_PATCHES, PATCH_SIZE),
#             sample_patches(test_im, NUM_PATCHES, PATCH_SIZE)[:,:30,:,:]), axis=0)
trainset = np.concatenate((sample_patches(flas[0], NUM_PATCHES, PATCH_SIZE),
            sample_patches(flas[1], NUM_PATCHES, PATCH_SIZE)[:,:30,:,:]), axis=0)
# testset = sample_patches(test_im, NUM_PATCHES, PATCH_SIZE)

trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
# testloader = DataLoader(
#     testset,
#     batch_size=BATCH_SIZE,
#     shuffle=True
# )

# utility functions
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

class LinearAutoencoder(nn.Module):
    """
    Starter version of the model. The model is linear, and takes in HxWxD, down to number of neurons.
    """
    def __init__(self, in_features=108000, num_neurons=100):
        super(LinearAutoencoder, self).__init__()
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
print(net)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

def train(net, trainloader, NUM_EPOCHS):
    train_loss = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for batch in trainloader:
            batch = batch.to(device)
            # batch = batch.view(batch.size(0), -1) # if linear autoencoder
            # if conv autoencoder
            optimizer.zero_grad()
            outputs = net(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch+1, NUM_EPOCHS, loss))
        # if epoch % 5 == 0:
        #     save_decoded_image(outputs.cpu().data, epoch, IM_PATH)
    return train_loss

def viz_image_reconstruction(net, testloader):
     for batch in testloader:
        batch = batch.to(device)
        batch = batch.view(batch.size(0), -1)
        outputs = net(batch)
        outputs = outputs.view(outputs.size(0), 1, 28, 28).cpu().data
        save_image(outputs, 'autoencoder_reconstruction.png')
        break

# get the computation device
now = datetime.datetime.now()
device = get_device()
print(device)
# load the neural network onto the device
net.to(device)
# train the network
train_loss = train(net, trainloader, NUM_EPOCHS)
torch.save({
'state_dict': net.state_dict(),
'optimizer': optimizer.state_dict(),
'loss': train_loss
}, 'checkpoint_{}.t7'.format(now))
plt.figure()
plt.plot(train_loss)
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('autoencoder_loss.png')
# test the network
# viz_image_reconstruction(net, testloader)
