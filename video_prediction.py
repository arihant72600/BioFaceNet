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
parser.add_argument('--top_pad')
parser.add_argument('--left_pad')
parser.add_argument('--size')

args = parser.parse_args()


l = int(args.top_pad)
r = int(args.left_pad)
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


fourcc = cv2.VideoWriter_fourcc(*'MP42')
video_out = cv2.VideoWriter('./original.avi', fourcc, frameRate, (64, 64))
video_recon = cv2.VideoWriter('./reconstruction.avi',
                              fourcc, frameRate, (64, 64))
video_blood = cv2.VideoWriter('./blood.avi', fourcc, frameRate, (64, 64))
video_mel = cv2.VideoWriter('./melanin.avi', fourcc, frameRate, (64, 64))
video_shad = cv2.VideoWriter('./shading.avi', fourcc, frameRate, (64, 64))
video_spec = cv2.VideoWriter('./specular.avi', fourcc, frameRate, (64, 64))


x = 1
while(cap.isOpened()):
    frameId = cap.get(1)  # current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    frame = frame[l:l+s, r:r+s, :]
    frame = cv2.resize(frame, (64, 64))
    video_out.write(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.transpose(frame, (2, 0, 1))
    frame = toLinearSpace(frame / 255)
    frame[0] -= meanPixel[0]
    frame[1] -= meanPixel[1]
    frame[2] -= meanPixel[2]
    images.append(frame)
    x += 1


video_out.release()

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

    encoder.eval()

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
            video_recon.write(cv2.cvtColor(
                (npimage * 255).astype('uint8'), cv2.COLOR_BGR2RGB))

            nptruth = np.array(truth)
            nptruth = np.transpose(nptruth, (1, 2, 0))

            spec = np.array(spec)
            spec = np.transpose(spec, (1, 2, 0))
            video_spec.write(cv2.cvtColor(
                (spec * 255).astype('uint8'), cv2.COLOR_BGR2RGB))

            fig = plt.figure()
            plt.imshow(npimage)

            fig.savefig(
                f"video-results/{str(index).zfill(6)}-reconstruction.png")

            fig = plt.figure()
            plt.imshow(nptruth)
            fig.savefig(f"video-results/{str(index).zfill(6)}-original.png")

            fig = plt.figure()
            plt.imshow(shade, cmap='gray')
            video_shad.write(
                cv2.cvtColor((np.array(shade) * 255).astype('uint8'), cv2.COLOR_GRAY2BGR))
            fig.savefig(f"video-results/{str(index).zfill(6)}-shading.png")

            fig = plt.figure()
            plt.imshow(spec)
            fig.savefig(
                f"video-results/{str(index).zfill(6)}-specularities.png")

            fig = plt.figure()
            plt.imshow(blood[0].to('cpu'))
            video_blood.write(
                cv2.cvtColor((np.array(blood[0]) * 255).astype('uint8'), cv2.COLOR_GRAY2BGR))
            fig.savefig(f"video-results/{str(index).zfill(6)}-blood.png")

            fig = plt.figure()
            plt.imshow(mel[0].to('cpu'))
            video_mel.write(
                cv2.cvtColor((np.array(mel[0]) * 255).astype('uint8'), cv2.COLOR_GRAY2BGR))
            fig.savefig(f"video-results/{str(index).zfill(6)}-melanin.png")

            index += 1

video_recon.release()
video_blood.release()
video_mel.release()
video_recon.release()
video_shad.release()
