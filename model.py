import torch
import torch.nn as nn
import torch.nn.functional as F

channels = [3, 32, 64, 128, 256, 512]

class Unet(nn.Module):
    def __init__(self, n_components = 2, lightVectorSize=15):
        super(Unet, self).__init__()

        self.convolutions = nn.ModuleList()
        self.encoderBatchnorms = nn.ModuleList()
        size = 64
        for i in range(1, len(channels)):
            self.convolutions.append(
                nn.Conv2d(channels[i - 1], channels[i], 3, padding=1))
            self.encoderBatchnorms.append(nn.BatchNorm2d(channels[i]))

            self.convolutions.append(
                nn.Conv2d(channels[i], channels[i], 3, padding=1))
            self.encoderBatchnorms.append(nn.BatchNorm2d(channels[i]))

            self.convolutions.append(
                nn.Conv2d(channels[i], channels[i], 3, padding=1))
            self.encoderBatchnorms.append(nn.BatchNorm2d(channels[i]))

            if i != len(channels) - 1:
                size //= 2

        self.low_resolution = size

        self.fc1 = nn.Linear(channels[-1] * size * size, channels[-1])
        self.batchnorm1 = nn.BatchNorm1d(channels[-1])
        self.fc2 = nn.Linear(channels[-1], channels[-1])
        self.batchnorm2 = nn.BatchNorm1d(channels[-1])
        self.fc3 = nn.Linear(channels[-1], lightVectorSize + n_components)

        self.decoderConvolutions = nn.ModuleList()
        self.decoderBatchnorms = nn.ModuleList()

        for _ in range(4):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            for i in reversed(range(1, len(channels) - 1)):
                size *= 2

                convs.append(
                    nn.Conv2d(channels[i] + channels[i + 1], channels[i], 3, padding=1))
                bns.append(nn.BatchNorm2d(channels[i]))
                convs.append(nn.Conv2d(channels[i], channels[i], 3, padding=1))
                bns.append(nn.BatchNorm2d(channels[i]))
                convs.append(nn.Conv2d(channels[i], channels[i], 3, padding=1))
                bns.append(nn.BatchNorm2d(channels[i]))

            convs.append(nn.Conv2d(channels[1], 1, 3, padding=1))

            self.decoderConvolutions.append(convs)
            self.decoderBatchnorms.append(bns)

    def forward(self, x):
        image = x # (B, 3, 64, 64)

        skipValues = []

        ########### Encoding ###########
        for convIndex in range(len(self.convolutions)):
            image = self.convolutions[convIndex](image)
            image = self.encoderBatchnorms[convIndex](image)
            image = F.relu(image)

            if convIndex % 3 == 2 and convIndex != len(self.convolutions) - 1:
                skipValues.append(torch.clone(image))
                image = F.max_pool2d(image, 2)

        skipValues.reverse()

        ########### Fully Connected Layer ###########
        # (B, channels[-1], low_resolution, low_resolution) -> (B, channels[-1], * low_resolution *low_resolution)
        lighting = torch.reshape(
            image, (-1, self.low_resolution * self.low_resolution * channels[-1]))
        lighting = self.fc1(lighting)
        lighting = self.batchnorm1(lighting)
        lighting = F.relu(lighting)
        lighting = self.fc2(lighting)
        lighting = self.batchnorm2(lighting)
        lighting = F.relu(lighting)
        lighting = self.fc3(lighting)

        features = []

        ########### Decoding ###########

        for out in range(4):
            feature = torch.clone(image)

            for i in range(len(self.decoderConvolutions[out]) - 1):
                if i % 3 == 0:
                    feature = F.interpolate(feature, scale_factor=2)
                    feature = torch.cat((feature, skipValues[i // 3]), 1)

                feature = self.decoderConvolutions[out][i](feature)
                feature = self.decoderBatchnorms[out][i](feature)
                feature = F.relu(feature)

            feature = self.decoderConvolutions[out][-1](feature)
            features.append(feature)

        return lighting, features

        