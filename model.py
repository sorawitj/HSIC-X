import collections

import torch
from joblib import delayed, Parallel
import torch.nn as nn
from sklearn.multioutput import MultiOutputRegressor
from torch.optim.lr_scheduler import StepLR
from statsmodels.tools.tools import add_constant
from collections import OrderedDict

from torch.utils.data import TensorDataset

from hsic import HSIC
import numpy as np
import pytorch_lightning as pl
from sklearn.linear_model import Ridge, LinearRegression

from kernel import Kernels, CategoryKernel, RBFKernel
from utils import dhsic_test, get_med_sigma, med_sigma, optimize_pval, radial_torch, radial, to_torch, fit_restart
from statsmodels.sandbox.regression.gmm import LinearIVGMM


class HSICModel(pl.LightningModule):

    def __init__(self, input_dim,
                 lmd=0, lr=1e-4,
                 kernel_e=None,
                 kernel_z=None):
        super(HSICModel, self).__init__()
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


class MedianHeuristic(pl.Callback):

    def __init__(self):
        self.epoch = 0

    def on_train_start(self, trainer, pl_module):
        data = trainer.train_dataloader.dataset.datasets.tensors
        s_e = get_med_sigma(pl_module, data, s_z=False)
        if pl_module.kernels.e.__class__ == RBFKernel:
            pl_module.kernels.e.set_kernel_param(s_e)
        if pl_module.kernels.z.__class__ == RBFKernel:
            s_z = med_sigma(data[2])
            pl_module.kernels.z.set_kernel_param(s_z)

        # print("update s_e: {}".format(pl_module.kernels.e.lengthscale))

    def on_train_epoch_end(self, trainer, pl_module):
        if self.epoch < 1000:
            data = trainer.train_dataloader.dataset.datasets.tensors
            s_e = get_med_sigma(pl_module, data)
            if pl_module.kernels.e.__class__ == RBFKernel:
                pl_module.kernels.e.set_kernel_param(s_e)
        self.epoch += 1
        # print("update s_e: {}".format(pl_module.kernels.e.lengthscale))


class PvalueLog(pl.Callback):
    def __init__(self, every_epoch=False, B=100):
        self.every_epoch = every_epoch
        self.B = B
        self.p_value = None

    def print_pval(self, trainer, pl_module):
        X, Y, Z = trainer.train_dataloader.dataset.datasets.tensors
        # train with HSIC loss
        res = (Y - pl_module(X)).detach().numpy()
        if pl_module.kernels.z.__class__ == CategoryKernel:
            kernel = ['gaussian', 'discrete']
        else:
            kernel = ['gaussian', 'gaussian']
        p_value = dhsic_test(res, Z.detach().numpy(), kernel=kernel, B=self.B)

        print("p-value: {}".format(p_value))
        return p_value

    def on_train_end(self, trainer, pl_module):
        p_value = self.print_pval(trainer, pl_module)
        self.p_value = p_value

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
                 kernel_e=None,
                 kernel_z=None,
                 bias=False):
        self.bias = bias
        super(LinearModel, self).__init__(input_dim, lmd, lr, kernel_e, kernel_z)

    def initialize(self):
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 1, bias=self.bias),
        )


class PolyModel(HSICModel):
    def __init__(self, input_dim, degree=2,
                 lmd=0, lr=1e-4,
                 kernel_e=None,
                 kernel_z=None,
                 bias=True):

        self.degree = degree
        self.bias = bias

        super(PolyModel, self).__init__(input_dim, lmd, lr, kernel_e, kernel_z)

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


class RadialModel(HSICModel):
    def __init__(self, input_dim, num_basis=20, data_limits=(-5, 5),
                 lmd=0, lr=1e-4,
                 kernel_e=None,
                 kernel_z=None,
                 bias=True):

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


class NonlinearModel(HSICModel):
    def __init__(self, input_dim,
                 lmd=0, lr=1e-4,
                 kernel_e=None,
                 kernel_z=None):
        super(NonlinearModel, self).__init__(input_dim, lmd, lr, kernel_e, kernel_z)

    def initialize(self):
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1),
        )


# class CNNModel(HSICModel):
#     def __init__(self, input_dim,
#                  lmd=0, gamma=0,
#                  lr=1e-4,
#                  kernel_e=None,
#                  kernel_z=None):
#         super(CNNModel, self).__init__(input_dim, lmd, gamma, lr, kernel_e, kernel_z)
#
#     def initialize(self):
#         self.cnn = DefaultCNN(cuda=False)
#
#     def forward(self, X):
#         return self.cnn(X)
#
#     def load_state_dict(self, module, **kwargs):
#         self.cnn.load_state_dict(module.cnn.state_dict())


class PredPolyRidge(object):
    def __init__(self, degree, alpha=0.2, bias=False):
        self.degree = degree
        self.bias = bias
        self.alpha = alpha
        if alpha == 0:
            self.reg = LinearRegression(fit_intercept=False)
        else:
            self.reg = Ridge(fit_intercept=False, alpha=self.alpha)

    def make_features(self, X):
        """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
        if X.ndim == 1:
            X = X[:, np.newaxis]
        # add a column of one if bias is True
        if self.bias:
            start = 0
        else:
            start = 1

        return np.concatenate([X ** i for i in range(start, self.degree + 1)], 1)

    def fit(self, X, Y):
        # 2SLS
        self.reg.fit(self.make_features(X), Y)

    def predict(self, X_test):
        y_pred = self.reg.predict(self.make_features(X_test))

        return y_pred


class PredRadialRidge(object):

    def __init__(self, num_basis, data_limits=(-5, 5), alpha=0.2, bias=False):
        self.num_basis = num_basis
        self.data_limits = data_limits
        self.bias = bias
        self.alpha = alpha
        self.reg = Ridge(random_state=123, fit_intercept=False, alpha=self.alpha)

    def make_features(self, X):
        """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
        if X.ndim == 1:
            X = X[:, np.newaxis]
        # add a column of one if bias is True
        phi = radial(X, num_basis=self.num_basis, data_limits=self.data_limits)
        if self.bias:
            phi = add_constant(phi)

        return phi

    def fit(self, X, Y):
        # 2SLS
        self.reg.fit(self.make_features(X), Y)

    def predict(self, X_test):
        y_pred = self.reg.predict(self.make_features(X_test))

        return y_pred


class Poly2SLS(object):
    def __init__(self, degree, bias=False):
        self.degree = degree
        self.bias = bias

    def make_features(self, X, bias):
        """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
        if X.ndim == 1:
            X = X[:, np.newaxis]
        # add a column of one if bias is True
        if bias:
            start = 0
        else:
            start = 1

        return np.concatenate([X ** i for i in range(start, self.degree + 1)], 1)

    def fit(self, X, Y, Z):
        gmm = LinearIVGMM(Y, self.make_features(X, False), self.make_features(Z, False))
        self.gmm_res = gmm.fit()

    def predict(self, X_test):
        y_2sls = self.gmm_res.predict(self.make_features(X_test, False))

        return y_2sls


class Radial2SLS(object):
    def __init__(self, num_basis, data_limits=(-5, 5), bias=False):
        self.num_basis = num_basis
        self.data_limits = data_limits
        self.bias = bias

    def make_features(self, X):
        """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
        if X.ndim == 1:
            X = X[:, np.newaxis]
        # add a column of one if bias is True
        phi = radial(X, num_basis=self.num_basis, data_limits=self.data_limits)
        if self.bias:
            phi = add_constant(phi)

        return phi

    def fit(self, X, Y, Z):
        gmm = LinearIVGMM(Y, self.make_features(X), self.make_features(Z))
        self.gmm_res = gmm.fit()

    def predict(self, X_test):
        y_2sls = self.gmm_res.predict(self.make_features(X_test))

        return y_2sls


class Radial2SLSRidge(object):
    def __init__(self, num_basis, data_limits=(-5, 5), alpha=0.2, bias=False):
        self.num_basis = num_basis
        self.data_limits = data_limits
        self.alpha = alpha
        self.bias = bias
        self.reg1 = MultiOutputRegressor(Ridge(random_state=123, fit_intercept=False, alpha=self.alpha))
        # self.reg1 = LinearRegression(fit_intercept=False)
        self.reg2 = Ridge(random_state=123, fit_intercept=False, alpha=self.alpha)
        # self.reg2 = LinearRegression(fit_intercept=False)

    def make_features(self, X):
        """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
        if X.ndim == 1:
            X = X[:, np.newaxis]
        # add a column of one if bias is True
        phi = radial(X, num_basis=self.num_basis, data_limits=self.data_limits)
        if self.bias:
            phi = add_constant(phi)

        return phi

    def fit(self, X, Y, Z):
        # 2SLS
        self.reg1.fit(self.make_features(Z), self.make_features(X))
        X_hat = self.reg1.predict(self.make_features(Z))
        self.reg2.fit(X_hat, Y)

    def predict(self, X_test):
        y_2sls = self.reg2.predict(self.make_features(X_test))

        return y_2sls


class PolySLSRidge(object):
    def __init__(self, degree, alpha=0.2, bias=False):
        self.degree = degree
        self.alpha = alpha
        self.bias = bias
        self.reg1 = MultiOutputRegressor(Ridge(random_state=123, fit_intercept=False, alpha=self.alpha))
        # self.reg1 = LinearRegression(fit_intercept=False)
        self.reg2 = Ridge(random_state=123, fit_intercept=False, alpha=self.alpha)
        # self.reg2 = LinearRegression(fit_intercept=False)

    def make_features(self, X):
        """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
        if X.ndim == 1:
            X = X[:, np.newaxis]
        # add a column of one if bias is True
        if self.bias:
            start = 0
        else:
            start = 1

        return np.concatenate([X ** i for i in range(start, self.degree + 1)], 1)

    def fit(self, X, Y, Z):
        # 2SLS
        self.reg1.fit(self.make_features(Z), self.make_features(X))
        X_hat = self.reg1.predict(self.make_features(Z))
        self.reg2.fit(X_hat, Y)

    def predict(self, X_test):
        y_2sls = self.reg2.predict(self.make_features(X_test))

        return y_2sls


def invert_test(X, Y, Z, test, param):
    res = Y - param * X
    pval, stat = test(res, Z)
    return pval, stat


class ConfIntHSIC(object):
    def __init__(self, kernels=('gaussian', 'gaussian'), method='gamma', B=None):
        self.kernels = kernels
        self.method = method
        self.B = B
        self.test = lambda R, Z: dhsic_test(R, Z, kernels, method=method, B=B, statistics=True)

    def fit(self, param_range, X, Y, Z):
        ret = Parallel(n_jobs=-1)(delayed(invert_test)(X=X, Y=Y, Z=Z, test=self.test, param=p) for p in param_range)

        return ret


def train_HSIC_IV(hsic_net, config, X, Y, Z, verbose=True):
    batch_size = config['batch_size']
    max_epoch = config['max_epoch']
    num_restart = config['num_restart']
    trainloader = torch.utils.data.DataLoader(TensorDataset(to_torch(X), to_torch(Y), to_torch(Z)),
                                              batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    se_callback = MedianHeuristic()
    pval_callback = PvalueLog()

    hsic_net = fit_restart(trainloader, hsic_net, pval_callback, max_epoch, se_callback,
                           num_restart=num_restart, verbose=verbose)

    return hsic_net

def train_mse(mse_net, config, X, Y, Z):
    batch_size = config['batch_size']
    max_epoch = config['max_epoch']
    trainloader = torch.utils.data.DataLoader(TensorDataset(to_torch(X), to_torch(Y), to_torch(Z)),
                                              batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    trainer = pl.Trainer(max_epochs=max_epoch, enable_checkpointing=False,
                         enable_model_summary=False, enable_progress_bar=False)
    trainer.fit(mse_net, trainloader)

    return mse_net