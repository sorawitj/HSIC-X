import numpy as np
import torch
from torch.utils.data import TensorDataset
import pytorch_lightning as pl
from tqdm import tqdm

from helpers.hsic import dhsic_test
from helpers.utils import to_torch, get_med_sigma
from models.callbacks import MedianHeuristic, PvalueLog, optimize_pval


def fit_restart(trainloader, hsic_net, pval_callback, max_epochs, se_callback=None, num_restart=10, alpha=0.05,
                verbose=True):
# fit an HSIC-X model with the restarting heuristic
    max_pval = 0.0
    best_model = None
    for i in range(num_restart):
        # only restart params after the first iterations
        if i > 0:
            print("restart")
            hsic_net.initialize()
        trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[se_callback, pval_callback], enable_checkpointing=False,
                             enable_model_summary=False, log_every_n_steps=100, enable_progress_bar=verbose)
        trainer.fit(hsic_net, trainloader)
        pval = pval_callback.p_value
        if pval < alpha:
            if pval >= max_pval:
                best_model = hsic_net.layers.state_dict().copy()
                max_pval = pval
            print("iteration {}, reject the test!".format(i + 1))
        else:
            print("iteration {}, accept the test!".format(i + 1))
            best_model = hsic_net.layers.state_dict()
            break

    hsic_net.load_state_dict(best_model)

    return hsic_net


# a helper function to train an HSIC-X model
def train_HSIC_IV(hsic_net, config, X, Y, Z, verbose=True):
    """
    @param hsic_net: a model
    @param config: a dictionary containing 'batch_size', 'max_epoch', and 'num_restart'
    @param X: predictors (treatments)
    @param Y: response
    @param Z: instruments
    @param verbose: True/False
    """
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


# a helper function to train an MSE model
def train_mse(mse_net, config, X, Y, Z):
    """
    @param mse_net: a model
    @param config: a dictionary containing 'batch_size', and 'max_epoch'
    @param X: predictors (treatments)
    @param Y: response
    @param Z: instruments
    """
    batch_size = config['batch_size']
    max_epoch = config['max_epoch']
    trainloader = torch.utils.data.DataLoader(TensorDataset(to_torch(X), to_torch(Y), to_torch(Z)),
                                              batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    trainer = pl.Trainer(max_epochs=max_epoch, enable_checkpointing=False,
                         enable_model_summary=False, enable_progress_bar=False)
    trainer.fit(mse_net, trainloader)

    return mse_net


def invert_test(X, Y, Z, test, param):
    res = Y - param * X
    pval, stat = test(res, Z)
    return pval, stat


def get_pvalue_landscape(hsic_net, data):
    param_grid = torch.linspace(0, 5, 40)
    hsic_net.layers[0].bias.data = torch.Tensor([0.])
    p_value = []
    med_se = []
    for w in tqdm(param_grid):
        hsic_net.layers[0].weight.data = torch.Tensor([[w]])
        x, y, z = data
        x = x.view(x.size(0), -1)
        y_hat = hsic_net(x)
        res = y - y_hat
        bw_e = np.sqrt(hsic_net.s_e / 2)
        bw_z = np.sqrt(hsic_net.s_z / 2)
        p_value += [dhsic_test(res.detach().numpy(), z.detach().numpy(), bw_e, bw_z, B=100)]
        med_se += [get_med_sigma(hsic_net, data)]

    return param_grid, p_value, med_se


def get_loss_landscape(hsic_net, data, grid_size=50):
    param_grid = torch.linspace(-15, 20, grid_size)
    hsic_net.layers[0].bias.data = torch.Tensor([0.])
    loss = []
    grad = []
    med_se = []
    pval_se = []
    for w in tqdm(param_grid):
        # zero out grad!
        hsic_net.configure_optimizers()['optimizer'].zero_grad()

        hsic_net.layers[0].weight.data = torch.Tensor([[w]])
        inner_loss = hsic_net.get_loss(data)
        inner_loss.backward()
        inner_grad = hsic_net.layers[0].weight.grad
        loss += [inner_loss.detach().numpy().item()]
        grad += [inner_grad.detach().numpy().item()]
        med_se += [get_med_sigma(hsic_net, data)]
        pval_se += [optimize_pval(hsic_net, data, B=50)[0]]

    return param_grid, loss, grad, med_se, pval_se


def get_loss_landscape_poly(hsic_net, data, grid_size=50):
    X = np.linspace(-7, 7, grid_size)
    Y = np.linspace(-5, 15, grid_size)
    # hsic_net.layers[0].bias.data = torch.Tensor([0.])
    idx = np.mgrid[0:X.shape[0], 0:Y.shape[0]].reshape(2, -1).T
    param_grid = np.meshgrid(X, Y)

    loss = np.empty(shape=(X.shape[0], Y.shape[0]))
    grad_u = np.empty(shape=(X.shape[0], Y.shape[0]))
    grad_v = np.empty(shape=(X.shape[0], Y.shape[0]))
    for i, j in tqdm(idx):
        hsic_net.configure_optimizers()['optimizer'].zero_grad()
        hsic_net.layers[0].weight.data = torch.Tensor([[X[i], Y[j]]])
        # zero out grad!
        inner_loss = hsic_net.get_loss(data)
        inner_loss.backward()
        inner_grad = hsic_net.layers[0].weight.grad.detach().numpy()

        loss[i, j] += [inner_loss.detach().numpy().item()]
        grad_u[i, j] += [inner_grad[:, 0].item()]
        grad_v[i, j] += [inner_grad[:, 1].item()]

    return param_grid, loss, grad_u, grad_v
