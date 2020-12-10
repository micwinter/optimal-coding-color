import os
from torch.utils.data import DataLoader
import numpy as np
from spectral import *

from utilities import sample_patches


def data_initializer(patch_size=16,
    IM_PATH='data/landscapes_fla',
    train_list=[1],
    test_list=[2],
    batch_size=128,
    num_patches=1000,
    ):
    trainset = []
    testset= []
    # Load image as array
    for i in train_list:
        train_im = envi.open(os.path.join(IM_PATH, 'hyperspectral_%s.hdr' %i), os.path.join(IM_PATH, 'hyperspectral_%s.fla' %i))
        trainset.append(sample_patches(train_im, num_patches, patch_size))
    trainset = np.concatenate(trainset)

    for i in test_list:
        test_im = envi.open(os.path.join(IM_PATH,'hyperspectral_%s.hdr' %i), os.path.join(IM_PATH,'hyperspectral_%s.fla' %i))
        testset.append(sample_patches(test_im, num_patches, patch_size))
    testset = np.concatenate(testset)  

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=True
    )
    return trainloader, testloader

