import torch
import torch.nn.functional as F

from utils import to_torch


class Kernels(torch.nn.Module):
    def __init__(self, kernel_e, kernel_z):
        super().__init__()
        self.e = kernel_e
        self.z = kernel_z


class PTKGauss(torch.nn.Module):
    """
    Pytorch implementation of the isotropic Gaussian kernel.
    Parameterization is the same as in the density of the standard normal
    distribution. sigma2 is analogous to the variance.
    """

    def __init__(self, sigma):
        """
        sigma2: a number representing the squared bandwidth
        """
        super().__init__()
        assert sigma > 0, 'sigma2 must be > 0. Was %s' % str(sigma)
        # need to be a tensor to make it tunable with a torch optimizer
        self.sigma = sigma
        # _tunable_params = self.sigma2

    def set_kernel_param(self, sigma=None):
        self.sigma = to_torch(sigma)

    def forward(self, X):
        """
        Evaluate the Gaussian kernel on the two 2d Torch Tensors
        Parameters
        ----------
        X : n1 x d Torch Tensor
        Y : n2 x d Torch Tensor
        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        # sigma = self.sigma
        sumX2 = (X ** 2).sum(1, keepdim=True)
        # sumy2 = torch.sum(X ** 2, dim=1).view(1, -1)
        D2 = sumX2 - 2.0 * X.matmul(X.t()) + sumX2.t()
        D2 = D2.clamp(min=0)

        K = torch.exp(-D2 / (2 * self.sigma ** 2))
        return K


class CategoryKernel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        # Z must be category
        Z = Z.view(-1)
        Z_unique, Z = torch.unique(Z, return_inverse=True)
        Z = F.one_hot(Z, num_classes=Z_unique.size(0))
        ret = (Z @ Z.T).to(torch.float)
        return ret
