import scipy.io as sio
import torch
import numpy as np
from sklearn.decomposition import PCA
import torch.nn.functional as F

s0 = torch.Tensor([94.80, 104.80, 105.90, 96.80, 113.90, 125.60, 125.50, 121.30, 121.30,
                   113.50, 113.10, 110.80, 106.50, 108.80, 105.30, 104.40, 100.00, 96.00, 95.10, 89.10,
                   90.50, 90.30, 88.40, 84.00, 85.10, 81.90, 82.60, 84.90, 81.30, 71.90, 74.30, 76.40,
                   63.30])

s1 = torch.Tensor([43.40, 46.30, 43.90, 37.10, 36.70, 35.90, 32.60, 27.90, 24.30, 20.10,
                   16.20, 13.20, 8.60, 6.10, 4.20, 1.90, 0.00, -
                   1.60, -3.50, -3.50, -5.80, -7.20, -8.60,
                   -9.50, -10.90, -10.70, -12.00, -14.00, -13.60, -12.00, -13.30, -12.90, -10.60
                   ])
s2 = torch.Tensor([-1.1, -0.5, -0.7, -1.2, -2.6, -2.9, -2.8, -2.6, -2.6, -1.8, -1.5, -1.3,
                   -1.2, -1.0, -0.5, -0.3, 0.0, 0.2, 0.5, 2.1, 3.2, 4.1, 4.7, 5.1, 6.7, 7.3, 8.6, 9.8, 10.2,
                   8.3, 9.6, 8.5, 7.0])


class Decoder:
    def __init__(self, device, n_components=2, lightVectorSize=15):
        self.lightVectorSize = lightVectorSize

        illA = sio.loadmat("util/illumA.mat")
        illA = illA['illumA'][0][0]

        illA = illA / illA.sum()
        self.illA = torch.tensor(illA).to(device)

        self.s0 = s0.to(device)
        self.s1 = s1.to(device)
        self.s2 = s2.to(device)

        self.illF = torch.Tensor(sio.loadmat(
            'util/illF')['illF']).to(device)[0]
        self.illF = self.illF / torch.reshape(torch.sum(self.illF, 0), (1, 12))

        rgbData = sio.loadmat("util/rgbCMF.mat")
        cameraSensitivityData = np.array(list(np.array(rgbData['rgbCMF'][0])))

        pca = PCA(n_components)

        Y = np.transpose(cameraSensitivityData, (2, 0, 1))

        for camera in range(28):
            for channel in range(3):
                # should use max but doesn't matter since white balance divides
                Y[camera, channel] /= np.sum(Y[camera, channel])

        Y = np.resize(Y, (28, 99))

        pca.fit(Y)

        pcaComponents = pca.components_ * \
            np.resize(pca.explained_variance_ ** 0.5, (n_components, 1))
        # Done so that vector is on the same scale as matlab
        pcaComponents[1] *= -1

        self.pcaMeans = torch.reshape(torch.tensor(
            pca.mean_), (1, 99)).float().to(device)
        self.pcaComponents = torch.tensor(
            pcaComponents).permute(1, 0).float().to(device)

        Newskincolour = sio.loadmat('util/Newskincolour.mat')['Newskincolour']
        Newskincolour = Newskincolour.transpose((2, 0, 1))
        skinColor = torch.tensor(Newskincolour).to(device)
        self.skinColor = torch.reshape(skinColor, (1, 33, 256, 256))

        tmatrix = sio.loadmat("util/Tmatrix.mat")['Tmatrix']
        tmatrix = np.transpose(tmatrix, (2, 0, 1))
        tmatrix = torch.tensor(tmatrix).to(device)
        self.tmatrix = torch.reshape(tmatrix, (1, 9, 128, 128))
        self.txyx2rgb = torch.tensor([[3.2406, -1.537, -0.498],
                                      [-0.968, 1.8758, 0.0415],
                                      [0.0557, -0.204, 1.0570]]
                                     ).to(device)

    def chromacity(self, t):
        t = t * 21000
        t = t + 4000

        x1 = -4.6070 * (10 ** 9) / (t ** 3) + (2.9678 * 10 ** 6) / \
            (t ** 2) + (0.09911 * 10 ** 3) / t + 0.244063
        x2 = -2.0064 * (10 ** 9) / (t ** 3) + (1.9018 * 10 ** 6) / \
            (t ** 2) + (0.24748 * 10 ** 3) / t + 0.237040

        x = (t <= 7000) * x1 + (t > 7000) * x2

        y = -3 * x ** 2 + 2.87 * x - 0.275

        return x, y

    def illuminanceD(self, temp):
        x, y = self.chromacity(temp)

        m = 0.0241 + 0.2562 * x - 0.7341 * y
        m1 = (-1.3515 - 1.7703 * x + 5.9114 * y) / m
        m2 = (0.0300 - 31.4424 * x + 30.0717 * y) / m

        s = self.s0 + m1 * self.s1 + m2 * self.s2
        return s / torch.reshape(torch.sum(s, 1), (-1, 1))

    def __call__(self, lighting, features):
        lighting_parameters = lighting[:, :self.lightVectorSize]
        b = lighting[:, self.lightVectorSize:]

        mel, blood, shade, spec = features

        ########### Scaling ###########
        
        lighting_weights = lighting_parameters[:, :14]
        lighting_weights = F.softmax(lighting_weights, 1)
        weightA = lighting_weights[:, 0]
        weightA = torch.reshape(weightA, (-1, 1))
        weightD = lighting_weights[:, 1]
        weightD = torch.reshape(weightD, (-1, 1))
        fWeights = lighting_weights[:, 2:14]
        colorTemp = lighting_weights[:, 14]
        colorTemp = torch.reshape(colorTemp, (-1, 1))

        b = 6 * torch.sigmoid(b) - 3

        mel = -2 * torch.sigmoid(mel) + 1
        blood = -2 * torch.sigmoid(blood) + 1
        shade = torch.exp(shade)
        spec = torch.exp(spec)

        ########### Illumination ###########

        aLightVector = weightA * self.illA
        dLightVector = weightD * self.illuminanceD(colorTemp)
        fLightVector = F.linear(fWeights, self.illF)

        e = aLightVector + dLightVector + fLightVector
        eSums = torch.reshape(torch.sum(e, 1), (-1, 1))

        e = e / eSums

        S = F.linear(b, self.pcaComponents)
        S += self.pcaMeans

        S = F.relu(S)

        S = torch.reshape(S, (-1, 3, 33))

        lightColor = S * torch.reshape(e, (-1, 1, 33))
        lightColor = torch.sum(S, 2)

        ########### Specularities ###########

        spec = spec * torch.reshape(lightColor, (-1, 3, 1, 1))

        ########### Diffuse ###########

        bioPhysicalLayer = torch.cat((mel, blood), 1).permute((0, 2, 3, 1))
        skinColorGrid = self.skinColor.repeat(
            (bioPhysicalLayer.shape[0], 1, 1, 1))
        r_total = F.grid_sample(skinColorGrid, bioPhysicalLayer)

        spectra = r_total * torch.reshape(e, (-1, 33, 1, 1))
        spectra = torch.reshape(spectra, (-1, 1, 33, 64, 64))
        S = torch.reshape(S, (-1, 3, 33, 1, 1))

        diffuse = torch.sum(spectra * S, 2)
        diffuse = shade * diffuse

        raw = diffuse + spec

        ########### Camera Transformation ###########

        wb = raw / torch.reshape(lightColor, (-1, 3, 1, 1))

        tMatrixGrid = self.tmatrix.repeat((wb.shape[0], 1, 1, 1))
        bIndex = torch.reshape(b / 3, (-1, 1, 1, 2))

        ts = F.grid_sample(tMatrixGrid, bIndex)
        ts = torch.reshape(ts, (-1, 9, 1, 1))

        ix = ts[:, 0, :, :] * wb[:, 0, :, :] + ts[:, 3, :, :] * \
            wb[:, 1, :, :] + ts[:, 6, :, :] * wb[:, 2, :, :]
        iy = ts[:, 1, :, :] * wb[:, 0, :, :] + ts[:, 4, :, :] * \
            wb[:, 1, :, :] + ts[:, 7, :, :] * wb[:, 2, :, :]
        iz = ts[:, 2, :, :] * wb[:, 0, :, :] + ts[:, 5, :, :] * \
            wb[:, 1, :, :] + ts[:, 8, :, :] * wb[:, 2, :, :]

        ix = torch.reshape(ix, (-1, 1, 64, 64))
        iy = torch.reshape(iy, (-1, 1, 64, 64))
        iz = torch.reshape(iz, (-1, 1, 64, 64))

        xyz = torch.cat((ix, iy, iz), 1)

        xyz = xyz.permute((0, 2, 3, 1))
        rgb = F.linear(xyz, self.txyx2rgb)
        rgb = rgb.permute((0, 3, 1, 2))

        rgb = F.relu(rgb)

        shade = torch.reshape(shade, (-1, 64, 64))

        return rgb, shade, spec, blood, mel, b
