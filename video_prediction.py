import argparse
from random import randint
import time
import math

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import cv2

from model import Unet
from decoder import Decoder
from loss import Loss


parser = argparse.ArgumentParser()
parser.add_argument("--weights", help="Path to weights")
parser.add_argument("--cuda", help="Use cuda for acceleration",
                    action="store_true")
parser.add_argument('--left_pad')
parser.add_argument('--right-pad')
parser.add_argument('--size')

args = parser.parse_args()


l = int(args.left_pad)
r = int(args.right_pad)
s = int(args.size)

SAVED_WEIGHTS = args.weights


device = 'cuda' if args.cuda else 'cpu'

images = []


def toLinearSpace(image):
    image = np.power(((image + 0.055) / (1.055)), 2.4)

    return image


meanPixel = np.array([0.35064762, 0.21667774, 0.16786481])

videoFile = "newsom.mp4"
cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
frameRate = cap.get(5)  # frame rate
x = 1
while(cap.isOpened()):
    frameId = cap.get(1)  # current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        frame = frame[l:l+s, r:r+s, :]
        frame = cv2.resize(frame, (64, 64))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.transpose(frame, (2, 0, 1))
        frame = toLinearSpace(frame / 255)
        frame[0] -= meanPixel[0]
        frame[1] -= meanPixel[1]
        frame[2] -= meanPixel[2]
        images.append(frame)


imagesnp = np.array(images)

cap.release()


def gammaCorrection(x):
    x = (x ** (1/2.4) * (1.055)) - 0.055

    return x


print('Finished loading data')

image_tensor = torch.from_numpy(imagesnp).float().to(device)


init_dataset = TensorDataset(image_tensor)
dataloaders = {
    'val': DataLoader(init_dataset, batch_size=8, shuffle=False, num_workers=0)
}

n_components = 2
lightVectorSize = 15

encoder = Unet(n_components, lightVectorSize).to(device)
decoder = Decoder(device, n_components, lightVectorSize)

if SAVED_WEIGHTS is not None:
    encoder.load_state_dict(torch.load(SAVED_WEIGHTS))

index = 0

# Read from input file
for inputs in dataloaders['val']:
    inputs = inputs[0]
    inputs = inputs.to(device)

    image = inputs

    with torch.set_grad_enabled(False):
        images, shades, specs, bloods, mels, bs = decoder(*encoder(image))

        originalImage = torch.clone(inputs[:, :3, :, :]).to('cpu')
        originalImage[:, 0, :, :] += 0.35064762
        originalImage[:, 1, :, :] += 0.21667774
        originalImage[:, 2, :, :] += 0.16786481

        for image, truth, shade, spec, mel, blood in zip(images, originalImage, shades, specs, mels, bloods):
            image = gammaCorrection(image)
            truth = gammaCorrection(truth)

            npimage = np.array(image)
            npimage = np.transpose(npimage, (1, 2, 0))

            nptruth = np.array(truth)
            nptruth = np.transpose(nptruth, (1, 2, 0))

            spec = np.array(spec)
            spec = np.transpose(spec, (1, 2, 0))

            fig = plt.figure()
            plt.imshow(npimage)
            fig.savefig(
                f"video-results/{str(index).zfill(6)}-reconstruction.png")

            fig = plt.figure()
            plt.imshow(nptruth)
            fig.savefig(f"video-results/{str(index).zfill(6)}-original.png")

            fig = plt.figure()
            plt.imshow(shade, cmap='gray')
            fig.savefig(f"video-results/{str(index).zfill(6)}-shading.png")

            fig = plt.figure()
            plt.imshow(spec)
            fig.savefig(
                f"video-results/{str(index).zfill(6)}-specularities.png")

            fig = plt.figure()
            plt.imshow(blood[0].to('cpu'))
            fig.savefig(f"video-results/{str(index).zfill(6)}-blood.png")

            fig = plt.figure()
            plt.imshow(mel[0].to('cpu'))
            fig.savefig(f"video-results/{str(index).zfill(6)}-melanin.png")

            index += 1
