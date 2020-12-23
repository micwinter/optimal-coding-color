"""
This script explores as a baseline check whether PCA can recover the L, S, M cone wavelength selectivities from a hyperspectral image
"""
from spectral import *
import os
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import datetime
import h5py
from glob import glob

now = datetime.datetime.now() # current timestamp

IM_PATH = '/media/big_hdd/opt-color/hyperspec_ims'
im_list = glob(os.path.join(IM_PATH, '*.mat'))
ims = []
for im in im_list[:15]:
    try:
        temp = h5py.File(im)['rad'][:]
        if temp.shape == (31, 1392, 1300):
            ims.append(temp)
    except(OSError):
        print(f'Could not load {im}')
ims = np.stack(ims)

wavelength_i = h5py.File(im)['bands'][:]
wavelength_i = np.array([int(i) for i in wavelength_i])

# Collapse across pixels and across images
ims = ims.transpose(1,0,2,3)
ims = ims.reshape(ims.shape[0], -1)

pca = PCA(n_components=4)
pca.fit(ims.T)


# Project to first three dimensions and plot results
comp = pca.components_
fig = plt.figure(figsize=(15,5))
plt.plot(comp.T)
locs, labels = plt.xticks()
plt.legend(['Photoreceptor 1', 'Photoreceptor 2', 'Photoreceptor 3', 'Photoreceptor 4'])
plt.xticks(np.arange(len(wavelength_i)), wavelength_i)
plt.ylabel('Weight Value')
plt.xlabel('Wavelength')
plt.title('PCA Projection to 4 Photoreceptors')
plt.savefig(f'pca_4_{now}.png', bbox_inches='tight')
plt.close()
# Skree plot
# import ipdb; ipdb.set_trace()
plt.plot(pca.explained_variance_ratio_)
plt.ylabel('PCA explained variance ratio')
plt.xlabel('Components')
plt.title('Explained Variance of PCA Components')
plt.savefig(f'pca_scree_4_{now}.png', bbox_inches='tight')
