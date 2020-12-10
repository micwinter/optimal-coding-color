# def load_flas(root=IM_PATH, list_of_files):
#     files = []
#     for file_ in list_of_files:
#         files.append(
#         envi.open(os.path.join(IM_PATH, '{}.hdr'.format(file_)),
#         os.path.join(IM_PATH, '{}.fla'.format(file_))))

#TODO: Create batches by randomly subsampling image
# Load image as array
# train_im = envi.open(os.path.join(IM_PATH, 'landscape01.hdr'), os.path.join(IM_PATH, 'landscape01.fla'))
# test_im = envi.open(os.path.join(IM_PATH,'landscape02.hdr'), os.path.join(IM_PATH,'landscape02.fla'))
# flas = load_flas(['landscape01', 'landscape02'])
# Get samples
# trainset = np.concatenate((sample_patches(train_im, NUM_PATCHES, PATCH_SIZE),
#             sample_patches(test_im, NUM_PATCHES, PATCH_SIZE)[:,:30,:,:]), axis=0)
# trainset = np.concatenate((sample_patches(flas[0], NUM_PATCHES, PATCH_SIZE),
#             sample_patches(flas[1], NUM_PATCHES, PATCH_SIZE)[:,:30,:,:]), axis=0)
# # testset = sample_patches(test_im, NUM_PATCHES, PATCH_SIZE)

# trainloader = DataLoader(
#     trainset,
#     batch_size=BATCH_SIZE,
#     shuffle=True
# )
# # testloader = DataLoader(
# #     testset,
# #     batch_size=BATCH_SIZE,
# #     shuffle=True
# # )

# # utility functions
# def get_device():
#     if torch.cuda.is_available():
#         device = 'cuda:0'
#     else:
#         device = 'cpu'
#     return device

# class LinearAutoencoder(nn.Module):
#     """
#     Starter version of the model. The model is linear, and takes in HxWxD, down to number of neurons.
#     """
#     def __init__(self, in_features=108000, num_neurons=100):
#         super(LinearAutoencoder, self).__init__()
#         # encoder
#         self.enc = nn.Linear(in_features=in_features, out_features=num_neurons)
#         # decoder
#         self.dec = nn.Linear(in_features=num_neurons, out_features=in_features)

#     def forward(self, x):
#         # encoder
#         x = F.relu(self.enc(x))
#         # decoder
#         x = F.relu(self.dec(x))
#         return x

# class ConvAutoencoder(nn.Module):
#     """
#     Second version of the model. The model is convolutional, and takes in HxWxD, down to number of neurons.
#     """
#     def __init__(self, in_features=30, latent_size=3):
#         super(ConvAutoencoder, self).__init__()
#         # encoder
#         # conv layer (depth from 30 --> 1), 3x3 kernels
#         self.enc = nn.Conv2d(in_features, latent_size, 3)
#         # decoder
#         self.dec = nn.ConvTranspose2d(latent_size, in_features, 3)

#     def forward(self, x):
#         # encoder
#         x = F.relu(self.enc(x))
#         # decoder
#         x = F.relu(self.dec(x))
#         # x = torch.sigmoid(self.dec(x)) # try with sigmoid or relu
#         return x

# # net = LinearAutoencoder()
# net = ConvAutoencoder()
# print(net)

# criterion = nn.MSELoss()
# optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)


# # get the computation device
# now = datetime.datetime.now()
# device = get_device()
# print(device)
# # load the neural network onto the device
# net.to(device)
# # train the network
# train_loss = train(net, trainloader, NUM_EPOCHS)
# torch.save({
# 'state_dict': net.state_dict(),
# 'optimizer': optimizer.state_dict(),
# 'loss': train_loss
# }, 'checkpoint_{}.t7'.format(now))
# plt.figure()
# plt.plot(train_loss)
# plt.title('Train Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.savefig('autoencoder_loss.png')
