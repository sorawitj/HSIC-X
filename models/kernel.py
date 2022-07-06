import torch
import torch.nn.functional as F

from helpers.utils import to_torch


class Kernels(torch.nn.Module):
    def __init__(self, kernel_e, kernel_z):
        super().__init__()
        self.e = kernel_e
        self.z = kernel_z


class RBFKernel(torch.nn.Module):

    def __init__(self, sigma):
        super().__init__()
        assert sigma > 0, 'sigma must be > 0. Current %s' % str(sigma)
        self.sigma = sigma

    def set_kernel_param(self, sigma=None):
        self.sigma = to_torch(sigma)

    def forward(self, X):
        sumX2 = (X ** 2).sum(1, keepdim=True)
        D2 = sumX2 - 2.0 * X.matmul(X.t()) + sumX2.t()
        D2 = D2.clamp(min=0)

        K = torch.exp(-D2 / (2 * self.sigma ** 2))
        return K


class CategoryKernel(torch.nn.Module):
    def __init__(self, one_hot=True):
        super().__init__()
        self.one_hot = one_hot

    def forward(self, Z):
        if self.one_hot:
            # Z must be category
            Z = Z.view(-1)
            Z_unique, Z = torch.unique(Z, return_inverse=True)
            Z = F.one_hot(Z, num_classes=Z_unique.size(0))
            ret = (Z @ Z.T).to(torch.float)
        else:
            print(Z.shape)
            ret = torch.nn.functional.cosine_similarity(Z[:, :, None], Z.t()[None, :, :])

        return ret


class ProductKernel3(torch.nn.Module):

    def __init__(self, k1, k2, k3, d1, d2, d3):
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3

    def forward(self, X):
        K1 = self.k1(X[:, self.d1])
        K2 = self.k2(X[:, self.d2])
        K3 = self.k3(X[:, self.d3])
        return K1 * K2 * K3


class ProductKernel2(torch.nn.Module):

    def __init__(self, k1, k2, d1, d2):
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.d1 = d1
        self.d2 = d2

    def forward(self, X):
        K1 = self.k1(X[:, self.d1])
        K2 = self.k2(X[:, self.d2])
        return K1 * K2
