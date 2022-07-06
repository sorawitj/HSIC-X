import numpy as np
import torch
import torch.nn as nn

from helpers.hsic import HSIC
from helpers.utils import to_torch
from models.hsicx import HSICXBase

class PartialHSICX(HSICXBase):
    def __init__(self, input_dim,
                 lmd=0, lr=1e-4,
                 kernel_e=None,
                 kernel_z=None,
                 bias=False,
                 num_E=2):
        self.bias = bias
        self.num_E = num_E
        super(PartialHSICX, self).__init__(input_dim, lmd, lr, kernel_e, kernel_z)

    def initialize(self):
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 1, bias=self.bias),
        )
        self.alpha_E = torch.nn.Embedding(self.num_E, 1)
        torch.nn.init.xavier_uniform_(self.beta)

    def forward(self, XZ):
        X, Z = XZ[:, :-1], XZ[:, [-1]]
        if X.__class__ == np.ndarray:
            X = to_torch(X)
        if X.dim() == 1:
            X = X[:, np.newaxis]

        alpha_E = self.alpha_E(Z.type(torch.LongTensor))

        pred = self.layers(X[:, 0]) + X[:, 1] * alpha_E.squeeze()

        return pred

    def get_loss(self, batch):
        x, y, z = batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)
        if self.lmd == -99:
            loss = self.MSE(y_hat, y)
        else:
            zx2 = torch.vstack([z, x[:, 1]]).T
            loss = self.lmd * self.MSE(y_hat, y) + HSIC(y - y_hat, zx2, self.kernels.e, self.kernels.z)
        return loss


class Partial_HSICX_CONT(HSICXBase):
    def __init__(self, input_dim,
                 lmd=0, lr=1e-4,
                 kernel_e=None,
                 kernel_z=None,
                 bias=False,
                 interact=False):
        self.bias = bias
        self.interact = interact
        super(Partial_HSICX_CONT, self).__init__(input_dim, lmd, lr, kernel_e, kernel_z)

    def initialize(self):
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 1, bias=self.bias),
        )

        self.g_layers = nn.Sequential(
            nn.Linear(self.input_dim + 1, 1, bias=self.bias),
        )

    def forward(self, XZ):
        # X, Z = XZ
        X, Z = XZ[:, :-1], XZ[:, [-1]]
        if X.__class__ == np.ndarray:
            X = to_torch(X)
        if X.dim() == 1:
            X = X[:, np.newaxis]
        if self.interact:
            Z = torch.hstack([Z, Z * X[:, [1]]])
        pred = self.layers(X[:, [0]]).flatten() + self.g_layers(Z).flatten()

        return pred

    def get_loss(self, batch):
        x, y, z = batch
        x = x.view(x.size(0), -1)
        # y_hat = self.forward((x, z))
        y_hat = self.forward(x)
        if self.lmd == -99:
            loss = self.MSE(y_hat, y)
        else:
            # zx2 = torch.vstack([z, x[:, 1]]).T
            loss = self.lmd * self.MSE(y_hat, y) + HSIC(y - y_hat, z, self.kernels.e, self.kernels.z)
        return loss


class Partial_HSICX_Lin(HSICXBase):
    def __init__(self, input_dim,
                 lmd=0, lr=1e-4,
                 kernel_e=None,
                 kernel_z=None,
                 bias=False,
                 num_E=2):
        self.bias = bias
        self.num_E = num_E
        super(Partial_HSICX_Lin, self).__init__(input_dim, lmd, lr, kernel_e, kernel_z)

    def initialize(self):
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 1, bias=self.bias),
        )
        self.alpha_E = torch.nn.Embedding(self.num_E, 1)
        self.alpha_EX = torch.nn.Embedding(self.num_E, self.input_dim)

    def forward(self, XZ):
        X, Z = XZ[:, :-1], XZ[:, [-1]]
        if X.__class__ == np.ndarray:
            X = to_torch(X)
        if X.dim() == 1:
            X = X[:, np.newaxis]

        alpha_EX = self.alpha_EX(Z.type(torch.LongTensor).flatten())
        alpha_E = self.alpha_E(Z.type(torch.LongTensor).flatten())

        pred = self.layers(X).flatten() + torch.einsum('ij,ij->i', alpha_EX, X).flatten() + alpha_E.flatten()

        return pred

    def get_loss(self, batch):
        x, y, z = batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)
        if self.lmd == -99:
            loss = self.MSE(y_hat, y)
        else:
            # zx2 = torch.vstack([z, x[:, 0]]).T
            loss = self.lmd * self.MSE(y_hat, y) + HSIC(y - y_hat, z, self.kernels.e, self.kernels.z)
        return loss
