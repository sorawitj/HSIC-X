import numpy as np
import torch
from scipy.spatial.distance import pdist
from tqdm import tqdm
import rpy2.robjects as robj
import rpy2.robjects.packages as packages

dHSIC = packages.importr("dHSIC")


def gen_data(f, n, iv_type='mean', debug=False):
    U = rnorm(n)
    e_y = 2 * rnorm(n)
    if iv_type == 'mean':
        Z = 1 + 2 * np.random.binomial(n=3, p=0.3, size=n)
        X = np.random.normal(Z - Z.mean(), 1) + 2 * U
        Y = f(X) + 8 * U + e_y  # linear
    elif iv_type == 'var':
        Z = 1 + 2 * np.random.binomial(n=3, p=0.3, size=n)
        X = np.random.normal(0, Z) + 2 * U  # variance
        Y = f(X) + 8 * U + e_y  # linear
    elif iv_type == 'diffcoef':
        Z = (-2) ** np.random.binomial(n=1, p=0.7, size=n)
        X = Z * U
        Y = f(X) - 4 * U + e_y  # linear
    else:
        raise Exception('Invalid iv_type {}'.format(iv_type))

    x_vis = np.linspace(X.min(), X.max(), 100)
    x_vis = np.round(x_vis, 2)
    if debug:
        return X, Y, Z, U, e_y, x_vis
    else:
        return X, Y, Z, x_vis


def med_sigma(x):
    if x.ndim == 1:
        x = x[:, np.newaxis]

    return 4 * np.sqrt(np.median(pdist(x, 'sqeuclidean')))


def get_med_sigma(hsic_net, data, s_z=False):
    X, Y, Z = data
    res = (Y - hsic_net(X)).detach().numpy()
    if s_z:
        return med_sigma(res), med_sigma(Z)
    else:
        return med_sigma(res)


def rnorm(n):
    return np.random.normal(size=n)


def to_r(x):
    return robj.FloatVector(x)


def to_torch(arr):
    return torch.from_numpy(np.array(arr).astype(np.float32))


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


def optimize_pval(pl_module, data, B=20):
    X, Y, Z = data
    se_grid = np.logspace(np.log10(1e-2), np.log10(1e3), 6, base=10)
    pvals = []
    for se in se_grid:
        res = (Y - pl_module(X)).detach().numpy()
        bw_e = se
        bw_z = pl_module.kernel_z.lengthscale
        pvals += [dhsic_test(res, Z.detach().numpy(), bw_e, bw_z, B=B)]
    best_s_e = se_grid[np.argwhere(pvals == np.amin(pvals)).astype(int)].flatten()
    min_pvalue = np.min(pvals)

    return best_s_e, min_pvalue


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


def dhsic_test(X, Y, s_x=None, s_y=None, B=10):
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if s_x is None:
        pval = dHSIC.dhsic_test(X=to_r(X), Y=to_r(Y),
                                B=B).rx2("p.value")[0]
    else:
        pval = dHSIC.dhsic_test(X=to_r(X), Y=to_r(Y),
                                B=B,
                                kernel='gaussian.fixed',
                                bandwidth=to_r((s_x, s_y))).rx2("p.value")[0]
    return pval
