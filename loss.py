import torch


class Loss:
    def __init__(self, priorWeight, appearanceWeight, shadingWeight, sparsityWeight, size=64):
        self.priorWeight = priorWeight
        self.appearanceWeight = appearanceWeight
        self.shadingWeight = shadingWeight
        self.sparsityWeight = sparsityWeight
        self.size = size

    def __call__(self, rgb, shade, spec, b, shading, mask, x):
        scale = torch.sum(shade * shading * mask, (1, 2)) / \
            torch.sum(shade * shade * mask, (1, 2))

        scaledShading = torch.reshape(scale, (-1, 1, 1)) * shade
        alpha = (shading - scaledShading) * mask

        priorLoss = torch.sum(b ** 2) * self.priorWeight / x.shape[0]

        originalImage = torch.clone(x)
        originalImage[:, 0, :, :] += 0.35064762
        originalImage[:, 1, :, :] += 0.21667774
        originalImage[:, 2, :, :] += 0.16786481

        delta = ((originalImage - rgb) ** 2) * \
            torch.reshape(mask, (-1, 1, self.size, self.size))
        appearanceLoss = torch.sum(
            delta ** 2 / (self.size * self.size)) * 255 * 255 * self.appearanceWeight / x.shape[0]
        # Matlab implementation has image in (0 - 255) so we scale appropriately

        shadingLoss = torch.sum(alpha ** 2) * self.shadingWeight / x.shape[0]

        # Paper mentions divide by size of mask but not in implementation
        # computing on spec sparsity loss after lightColor transformation

        sparsityLoss = torch.sum(
            spec) * self.sparsityWeight / x.shape[0]  # Change to 1e-7

        return priorLoss, appearanceLoss, shadingLoss, sparsityLoss
