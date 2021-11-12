import torch
import torch.nn as nn
from pyro.nn import PyroModule
from sklearn.multioutput import MultiOutputRegressor
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from hsic import HSIC
import numpy as np
import pytorch_lightning as pl
from sklearn.linear_model import LinearRegression, Ridge
from pyro.contrib.gp.kernels import RBF, Linear, Polynomial

from kernel import RBFKernel
from utils import med_sigma, dhsic_test, get_med_sigma, to_torch, optimize_pval


class HSICModel(pl.LightningModule):

    def __init__(self, input_dim,
                 lmd=0, lr=1e-4,
                 kernels=None):
        super(HSICModel, self).__init__()
        self.lmd = lmd
        self.input_dim = input_dim
        self.lr = lr
        self.MSE = nn.MSELoss()
        self.layers = None
        self.kernels = RBFKernel() if kernels is None else kernels
        self.kernel_e, self.kernel_z = self.kernels.kernel_e, self.kernels.kernel_z

    def forward(self, X):
        if X.dim() == 1:
            X = X[:, np.newaxis]

        return self.layers(X).flatten()

    def get_loss(self, batch):
        x, y, z = batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)

        if self.lmd == 0:
            loss = self.MSE(y_hat, y)
        elif self.lmd == -99:
            loss = HSIC(y - y_hat, z, self.kernel_e, self.kernel_z)
        else:
            loss = self.MSE(y_hat, y) + self.lmd * HSIC(y - y_hat, z, self.kernel_e, self.kernel_z)
        return loss

    def training_step(self, batch, batch_idx):

        loss = self.get_loss(batch)
        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        learn_params = [p for n, p in self.named_parameters() if 'kernel' not in n]
        optimizer = torch.optim.Adam(learn_params, lr=self.lr)
        scheduler = StepLR(optimizer, step_size=200, gamma=0.2)

        return {'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'loss'}


class MedianHeuristic(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        data = trainer.train_dataloader.dataset.datasets.tensors
        s_e, s_z = get_med_sigma(pl_module, data, s_z=True)
        pl_module.kernel_e.lengthscale = to_torch(s_e)
        pl_module.kernel_z.lengthscale = to_torch(max(s_z, 0.001))
        print("update s_e: {}".format(pl_module.kernel_e.lengthscale))

    def on_train_epoch_end(self, trainer, pl_module):
        data = trainer.train_dataloader.dataset.datasets.tensors
        s_e = get_med_sigma(pl_module, data)
        pl_module.kernel_e.lengthscale = to_torch(s_e)
        print("update s_e: {}".format(pl_module.kernel_e.lengthscale))


class PvalueLog(pl.Callback):
    def __init__(self, every_epoch=False, B=100):
        self.every_epoch = every_epoch
        self.B = B

    def print_pval(self, trainer, pl_module):
        X, Y, Z = trainer.train_dataloader.dataset.datasets.tensors
        # train with HSIC loss
        res = (Y - pl_module(X)).detach().numpy()
        # bw_e = pl_module.kernel_e.lengthscale.item()
        # bw_z = pl_module.kernel_z.lengthscale.item()
        p_value = dhsic_test(res, Z.detach().numpy(), B=self.B)

        print("p-value: {}".format(p_value))

    def on_train_end(self, trainer, pl_module):
        self.print_pval(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.every_epoch:
            self.print_pval(trainer, pl_module)


class PvalueOptim(pl.Callback):

    def on_train_epoch_end(self, trainer, pl_module):
        data = trainer.train_dataloader.dataset.datasets.tensors
        best_s_e, min_pvalue = optimize_pval(data, pl_module)
        pl_module.s_e = best_s_e
        print("min p-value: {}, update s_e: {}".format(min_pvalue, best_s_e))


class LinearModel(HSICModel):
    def __init__(self, input_dim,
                 lmd=0, lr=1e-4,
                 kernels=None):
        super(LinearModel, self).__init__(input_dim, lmd, lr, kernels)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1),
        )


class PolyModel(HSICModel):
    def __init__(self, input_dim, degree=2,
                 lmd=0, lr=1e-4,
                 kernels=None,
                 bias=True):
        super(PolyModel, self).__init__(input_dim, lmd, lr, kernels)

        self.degree = degree
        self.bias = bias

        self.layers = nn.Sequential(
            nn.Linear(input_dim * degree, 1, bias=bias),
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


class NonlinearModel(HSICModel):
    def __init__(self, input_dim,
                 lmd=0, lr=1e-4,
                 kernels=None):
        super(NonlinearModel, self).__init__(input_dim, lmd, lr, kernels)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Sigmoid(),
            nn.Linear(128, 1),
        )


class Poly2SLS(object):
    def __init__(self, degree):
        self.degree = degree
        # self.reg1 = MultiOutputRegressor(Ridge(random_state=123))
        self.reg1 = LinearRegression(fit_intercept=False)
        # self.reg2 = Ridge(random_state=123, fit_intercept=True)
        self.reg2 = LinearRegression(fit_intercept=False)

    def make_features(self, X):
        """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
        if X.ndim == 1:
            X = X[:, np.newaxis]
        return np.concatenate([X ** i for i in range(1, self.degree + 1)], 1)

    def fit(self, X, Y, Z):
        # 2SLS
        self.reg1.fit(self.make_features(Z), self.make_features(X))
        X_hat = self.reg1.predict(self.make_features(Z))
        self.reg2.fit(X_hat, Y)

    def predict(self, X_test):
        y_2sls = self.reg2.predict(self.make_features(X_test))

        return y_2sls
