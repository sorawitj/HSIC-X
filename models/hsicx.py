import collections
from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from joblib import delayed, Parallel
from torch.optim.lr_scheduler import StepLR

from helpers.hsic import HSIC, dhsic_test
from helpers.trainer import invert_test
from helpers.utils import radial_torch, to_torch
from models.kernel import Kernels


class HSICXBase(pl.LightningModule):

    def __init__(self, input_dim,
                 lmd=0, lr=1e-4,
                 kernel_e=None,
                 kernel_z=None):
        """
        @param input_dim: predictor (treatment) dimensions
        @param lmd: regularization parameter for HSIC-X-pen
        @param lr: learning rate
        @param kernel_e: kernel for residuals
        @param kernel_z: kernel for instruments
        """
        super(HSICXBase, self).__init__()
        self.lmd = lmd
        self.input_dim = input_dim
        self.lr = lr
        self.MSE = nn.MSELoss()
        self.layers = None
        self.kernels = Kernels(kernel_e, kernel_z)
        self.initialize()

    def initialize(self):
        pass

    def forward(self, X):
        if X.__class__ == np.ndarray:
            X = to_torch(X)
        if X.dim() == 1:
            X = X[:, np.newaxis]

        return self.layers(X).flatten()

    def get_loss(self, batch):
        x, y, z = batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)
        if self.lmd == -99:
            loss = self.MSE(y_hat, y)
        else:
            loss = self.lmd * self.MSE(y_hat, y) + HSIC(y - y_hat, z, self.kernels.e, self.kernels.z)
        return loss

    def training_step(self, batch, batch_idx):

        loss = self.get_loss(batch)
        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        learn_params = [p for n, p in self.named_parameters() if 'kernel' not in n]
        optimizer = torch.optim.Adam(learn_params, lr=self.lr)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.8)

        return {'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'loss'}

    def load_state_dict(self, module, **kwargs):
        if module.__class__ == collections.OrderedDict:
            self.layers.load_state_dict(module)
        elif module.__class__ == np.ndarray:
            key = list(self.layers.state_dict().keys())[0]
            coef = to_torch(module.reshape(1, -1))
            new_state = OrderedDict([(key, coef)])
            self.layers.load_state_dict(new_state)
        else:
            self.layers.load_state_dict(module.layers.state_dict())


class LinearHSICX(HSICXBase):
    def __init__(self, input_dim,
                 lmd=0, lr=1e-4,
                 kernel_e=None,
                 kernel_z=None,
                 bias=False):
        """
        @param input_dim: predictor (treatment) dimensions
        @param lmd: regularization parameter for HSIC-X-pen
        @param lr: learning rate
        @param kernel_e: kernel for residuals
        @param kernel_z: kernel for instruments
        @param bias: include a bias term or not
        """
        self.bias = bias
        super(LinearHSICX, self).__init__(input_dim, lmd, lr, kernel_e, kernel_z)

    def initialize(self):
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 1, bias=self.bias),
        )


class PolyHSICX(HSICXBase):
    def __init__(self, input_dim, degree=2,
                 lmd=0, lr=1e-4,
                 kernel_e=None,
                 kernel_z=None,
                 bias=True):
        """
        @param input_dim: predictor (treatment) dimensions
        @param degree: polynomial's degree
        @param lmd: regularization parameter for HSIC-X-pen
        @param lr: learning rate
        @param kernel_e: kernel for residuals
        @param kernel_z: kernel for instruments
        @param bias: include a bias term or not
        """
        self.degree = degree
        self.bias = bias

        super(PolyHSICX, self).__init__(input_dim, lmd, lr, kernel_e, kernel_z)

    def initialize(self):
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim * self.degree, 1, bias=self.bias),
        )

    def make_features(self, X):
        """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
        if X.dim() == 1:
            X = X[:, np.newaxis]
        return torch.cat([X ** i for i in range(1, self.degree + 1)], 1)

    def forward(self, X):
        if X.dim() == 1:
            X = X[:, np.newaxis]

        polyX = self.make_features(X)

        return self.layers(polyX).flatten()


class RadialHSICX(HSICXBase):
    def __init__(self, input_dim, num_basis=20, data_limits=(-5, 5),
                 lmd=0, lr=1e-4,
                 kernel_e=None,
                 kernel_z=None,
                 bias=True):
        """
        @param input_dim: predictor (treatment) dimensions
        @param num_basis: number of radial basis functions
        @param data_limits: min and max to create a grid for the radial basis functions
        @param lmd: regularization parameter for HSIC-X-pen
        @param lr: learning rate
        @param kernel_e: kernel for residuals
        @param kernel_z: kernel for instruments
        @param bias: include a bias term or not
        """

        self.num_basis = num_basis
        self.data_limits = data_limits
        self.bias = bias

        super().__init__(input_dim, lmd, lr, kernel_e, kernel_z)

    def initialize(self):
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim * (self.num_basis + 2), 1, bias=self.bias),
        )

    def make_features(self, X):
        """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
        if X.dim() == 1:
            X = X[:, np.newaxis]
        return radial_torch(X, num_basis=self.num_basis, data_limits=self.data_limits)

    def forward(self, X):
        if X.dim() == 1:
            X = X[:, np.newaxis]

        polyX = self.make_features(X)

        return self.layers(polyX).flatten()


class NNHSICX(HSICXBase):
    def __init__(self, input_dim,
                 lmd=0, lr=1e-4,
                 kernel_e=None,
                 kernel_z=None):
        """
        @param input_dim: predictor (treatment) dimensions
        @param lmd: regularization parameter for HSIC-X-pen
        @param lr: learning rate
        @param kernel_e: kernel for residuals
        @param kernel_z: kernel for instruments
        """

        super(NNHSICX, self).__init__(input_dim, lmd, lr, kernel_e, kernel_z)

    def initialize(self):
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1),
        )


class ConfIntHSIC(object):
    def __init__(self, kernels=('gaussian', 'gaussian'), method='gamma', B=None):
        """
        @param kernels: kernels for the HSIC test
        @param method: 'gamma' -> gamma approximation, else permutation test
        @param B: number of permutations used if method != 'gamma'
        """
        self.kernels = kernels
        self.method = method
        self.B = B
        self.test = lambda R, Z: dhsic_test(R, Z, kernels, method=method, B=B, statistics=True)

    def fit(self, param_range, X, Y, Z):
        ret = Parallel(n_jobs=-1)(delayed(invert_test)(X=X, Y=Y, Z=Z, test=self.test, param=p) for p in param_range)

        return ret
