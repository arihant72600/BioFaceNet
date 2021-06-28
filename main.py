import argparse
from random import randint
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

from model import Unet
from decoder import Decoder
from loss import Loss


parser = argparse.ArgumentParser()
parser.add_argument("--weights", help="Path to weights")
parser.add_argument("--cuda", help="Use cuda for acceleration",
                    action="store_true")
args = parser.parse_args()

SAVED_WEIGHTS = args.weights


device = 'cuda' if args.cuda else 'cpu'

print("Loading Data")

images = h5py.File('data/lin_image_shading_mask_all_1.hdf5', 'r')['default']


def gammaCorrection(x):
    x = (x ** (1/2.4) * (1.055)) - 0.055

    return x


meanPixel = np.array([0.35064762, 0.21667774, 0.16786481])
datanp = np.array(images)

imagesnp = datanp[:, :3, :, :]
shadingnp = datanp[:, 3, :, :]
masknp = datanp[:, 4, :, :]

print('Finished loading data')

traindata_proportion = 0.9

image_tensor = torch.from_numpy(imagesnp).to(device)
shading_tensor = torch.from_numpy(shadingnp).to(device)
mask_tensor = torch.from_numpy(masknp).to(device)


init_dataset = TensorDataset(image_tensor, shading_tensor, mask_tensor)
lengths = [int(len(init_dataset)*traindata_proportion), len(init_dataset) -
           int(len(init_dataset)*traindata_proportion)]
subset_train, subset_val = random_split(init_dataset, lengths)

dataloaders = {
    'train': DataLoader(subset_train, batch_size=32, shuffle=True, num_workers=0),
    'val': DataLoader(subset_val, batch_size=8, shuffle=False, num_workers=0)
}

n_components = 2
lightVectorSize = 15

priorWeight = 1e-4
appearanceWeight = 1e-3
shadingWeight = 1e-5
sparsityWeight = 1e-7

encoder = Unet(n_components, lightVectorSize).to(device)
decoder = Decoder(device, n_components, lightVectorSize)
custom_loss = Loss(priorWeight, appearanceWeight,
                   shadingWeight, sparsityWeight)

if SAVED_WEIGHTS is not None:
    encoder.load_state_dict(torch.load(SAVED_WEIGHTS))

smallestLoss = 1e9
num_epochs = 100

optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-5, weight_decay=0)


for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    logs = {}

    # let every epoch go through one training cycle and one validation cycle
    # TRAINING AND THEN VALIDATION LOOP...
    for phase in ['train', 'val']:
        train_loss = 0
        prior_train_loss = 0
        appearance_train_loss = 0
        shading_train_loss = 0
        prior_train_loss = 0
        sparsity_train_loss = 0

        total = 0
        batch_idx = 0

        start_time = time.time()

        if phase == 'train':
            for param_group in optimizer.param_groups:
                print("LR", param_group['lr'])  # print out the learning rate
            encoder.train()  # Set model to training mode
        else:
            encoder.eval()   # Set model to evaluate mode

        for image, shading, mask in dataloaders[phase]:
            image = image.to(device)
            shading = shading.to(device)
            mask = mask.to(device)

            batch_idx += 1

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                rgb, shade, spec, blood, mel, b = decoder(*encoder(image))
                priorLoss, appearanceLoss, shadingLoss, sparsityLoss = custom_loss(
                    rgb, shade, spec, b, shading, mask, image)

                totalLoss = appearanceLoss + shadingLoss + priorLoss + sparsityLoss

                if phase == 'train':
                    totalLoss.backward()
                    optimizer.step()

                appearance_train_loss += appearanceLoss
                shading_train_loss += shadingLoss
                prior_train_loss += priorLoss
                sparsity_train_loss += sparsityLoss
                train_loss += totalLoss

        prefix = ''
        if phase == 'val':
            prefix = 'val_'
            if appearance_train_loss.item() < smallestLoss:
                torch.save(encoder.state_dict(), 'working/model.pth')
                print('Saving')
                smallestLoss = appearance_train_loss.item()

        logs[prefix + 'loss'] = train_loss.item()/(batch_idx)
        logs[prefix + 'appearance loss'] = appearance_train_loss.item() / \
            (batch_idx)
        logs[prefix + 'shading loss'] = shading_train_loss.item()/(batch_idx)
        logs[prefix + 'prior loss'] = prior_train_loss.item()/(batch_idx)
        logs[prefix + 'sparsity loss'] = sparsity_train_loss.item()/(batch_idx)

    with torch.set_grad_enabled(False):
        encoder.eval()
        randIndex = randint(0, len(subset_val) - 1)

        inputData = torch.reshape(subset_val[randIndex][0], (1, 3, 64, 64))

        rgb, shade, spec, blood, mel, b = decoder(*encoder(inputData))
        originalImage = torch.clone(inputData[0]).to('cpu')
        originalImage[0] += 0.35064762
        originalImage[1] += 0.21667774
        originalImage[2] += 0.16786481

        rgb = gammaCorrection(rgb[0]).to('cpu')
        originalImage = gammaCorrection(originalImage)

        rgb = np.array(rgb)
        rgb = np.transpose(rgb, (1, 2, 0))

        originalImage = np.array(originalImage)
        originalImage = np.transpose(originalImage, (1, 2, 0))

        spec = np.array(spec[0].to('cpu'))
        spec = np.transpose(spec, (1, 2, 0))

        print('saving figures')

        epochString = str(epoch).zfill(3)

        fig = plt.figure()
        plt.imshow(rgb)
        fig.savefig(f'training/epoch-{epochString}-reconstruction.png')

        fig = plt.figure()
        plt.imshow(originalImage)
        fig.savefig(f'training/epoch-{epochString}-original.png')

        fig = plt.figure()
        plt.imshow(shade[0].to('cpu'), cmap='gray')
        fig.savefig(f'training/epoch-{epochString}-shading.png')

        fig = plt.figure()
        plt.imshow(spec)
        fig.savefig(f'training/epoch-{epochString}-specular.png')

        fig = plt.figure()
        plt.imshow(blood[0][0].to('cpu'), cmap='gray')
        fig.savefig(f'training/epoch-{epochString}-blood.png')

        fig = plt.figure()
        plt.imshow(mel[0][0].to('cpu'), cmap='gray')
        fig.savefig(f'training/epoch-{epochString}-melanin.png')
