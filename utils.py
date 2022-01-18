import numpy as np
import torch
from scipy.spatial.distance import pdist
from tqdm import tqdm
import rpy2.robjects.packages as packages
import pytorch_lightning as pl
from rpy2.robjects import numpy2ri

numpy2ri.activate()

dHSIC = packages.importr("dHSIC")


def radial(x, num_basis=20, data_limits=(-5., 5.)):
    if x.ndim == 1:
        x = x[:, np.newaxis]
    "Radial basis constructed using exponentiated quadratic form."
    if num_basis > 1:
        centres = np.linspace(data_limits[0], data_limits[1], num_basis)
    else:
        centres = np.asarray([data_limits[0] / 2. + data_limits[1] / 2.])

    Phi = np.zeros((x.shape[0], 2 + num_basis))
    for i in range(num_basis):
        Phi[:, i:i + 1] = np.exp(-1 * ((x - centres[i]) / 1) ** 2)
    Phi[:, [num_basis]] = x
    Phi[:, [num_basis + 1]] = x ** 2
    return Phi


def radial_torch(x, num_basis=20, data_limits=(-5., 5.)):
    "Radial basis constructed using exponentiated quadratic form."
    centres = torch.linspace(data_limits[0], data_limits[1], num_basis)

    Phi = torch.zeros((x.shape[0], 2 + num_basis))
    for i in range(num_basis):
        Phi[:, i:i + 1] = torch.exp(-1 * ((x - centres[i]) / 1) ** 2)
    Phi[:, [num_basis]] = x
    Phi[:, [num_basis + 1]] = x ** 2
    return Phi


def gen_radial_fn(num_basis, data_limits=(-5, 5)):
    w = np.random.normal(0, 2, size=num_basis)
    w = np.concatenate([w, [1.5, -0.2]])
    return lambda x: radial(x, num_basis=num_basis, data_limits=data_limits) @ w


# plt.scatter(x, y)

def gen_data(f, n, iv_type='mean', alpha=1, var_effect=True, debug=False, oracle=False):
    U = rnorm(n)
    e_y = rnorm(n)
    e_x = rnorm(n)
    if oracle:
        gamma = 0
    else:
        gamma = 1
    if iv_type == 'mean':
        Z = 1 + 2 * np.random.binomial(n=3, p=0.3, size=n)
        X = np.random.normal(Z - Z.mean(), 1) + 2 * U
        Y = f(X) + 8 * U + e_y  # linear
    elif iv_type == 'var':
        Z = 1 + 2 * np.random.binomial(n=3, p=0.3, size=n)
        X = np.random.normal(0, Z) + 2 * U  # variance
        Y = f(X) + 8 * U + e_y  # linear
    elif iv_type == 'mix_Gaussian':
        # Z = (-2) ** np.random.binomial(n=1, p=0.5, size=n)
        Z = rnorm(n)
        if var_effect:
            X = Z * e_x + gamma * U + alpha * Z
        else:
            X = gamma * U + alpha * Z + e_x
        # X = Z * U + alpha * (U + Z)
        # Y = f(X) + 2 * U + e_y  # linear
        Y = f(X) - 4 * U + e_y  # linear
    elif iv_type == 'mix_Binary':
        Z = (-2) ** np.random.binomial(n=1, p=1 / 3, size=n)
        # Z = rnorm(n)
        if var_effect:
            X = Z * e_x + gamma * U + alpha * Z
        else:
            X = gamma * U + alpha * Z + e_x
        # X = Z * U + alpha * (U + Z)
        # Y = f(X) + 2 * U + e_y  # linear
        Y = f(X) - 4 * U + e_y  # linear
    else:
        raise Exception('Invalid iv_type {}'.format(iv_type))

    x_vis = np.linspace(X.min(), X.max(), 10000)
    x_vis = np.round(x_vis, 3)
    if debug:
        return X, Y, Z, U, e_y, x_vis
    else:
        return X, Y, Z, x_vis


def gen_data_multi(f, n, x_dim, z_dim, iv_type='Gaussian', alpha=1, oracle=False):
    # U = rnorm_d(n, u_dim)
    U = rnorm(n)
    # bUx = np.random.RandomState(0).uniform(-2, 2, size=(u_dim, x_dim))
    # bUy = np.random.RandomState(1).uniform(-6, 6, size=(u_dim,))
    e_y = rnorm(n)
    e_x = rnorm_d(n, x_dim)
    # rep_ex = np.repeat(e_x, z_dim, axis=1).reshape(n, x_dim, -1)
    # bZ1 = np.random.RandomState(2).uniform(-4, 4, size=(z_dim, x_dim))
    # bZ1 = np.random.RandomState(5).normal(0, 2, size=(z_dim, x_dim))
    bZ1 = np.ones(shape=(z_dim, x_dim))
    # bZ2 = np.random.RandomState(3).uniform(-4, 4, size=(z_dim, x_dim))
    # bZ2 = np.random.RandomState(10).normal(0, 2., size=(z_dim, x_dim))
    bZ2 = np.ones(shape=(z_dim, x_dim))
    # bZ2 = np.ones(shape=(z_dim, x_dim))

    if oracle:
        g = 0
    else:
        g = 1

    if iv_type == 'mix_Gaussian':
        Z = rnorm_d(n, z_dim)
    elif iv_type == 'mix_Binary':
        Z = (-2) ** np.random.binomial(1, p=np.repeat(1 / 3, z_dim), size=(n, z_dim))
    else:
        raise Exception('Invalid iv_type {}'.format(iv_type))

    X = e_x * (Z @ bZ2) + g * U[:, np.newaxis] + alpha * Z @ bZ1
    # X = g * Z @ bZ2 * U[:, np.newaxis] + alpha * Z @ bZ1 + e_x
    Y = f(X) - 4 * U + e_y  # linear

    return X, Y, Z


def med_sigma(x):
    if x.ndim == 1:
        x = x[:, np.newaxis]

    # return 4 * np.sqrt(np.median(pdist(x, 'sqeuclidean')))
    return np.sqrt(np.median(pdist(x, 'sqeuclidean')) * .5)
    # return 4 * np.sqrt(np.median(pdist(x, 'sqeuclidean')))


def get_med_sigma(hsic_net, data, s_z=False):
    X, Y, Z = data
    res = (Y - hsic_net(X)).detach().numpy()
    if s_z:
        return med_sigma(res), med_sigma(Z)
    else:
        return med_sigma(res)


def rnorm(n):
    return np.random.normal(size=n)


def rnorm_d(n, d):
    mean = np.zeros(shape=(d,))
    cov = np.eye(d)
    return np.random.multivariate_normal(mean, cov, size=n)


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


def dhsic_test(X, Y, kernel=["gaussian", "gaussian"], s_x=None, s_y=None, method="gamma", B=10):
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if s_x is None:
        pval = dHSIC.dhsic_test(X=X, Y=Y,
                                method=method, kernel=kernel,
                                B=B).rx2("p.value")[0]
    else:
        pval = dHSIC.dhsic_test(X=X, Y=Y,
                                B=B, method=method,
                                kernel='gaussian.fixed',
                                bandwidth=(s_x, s_y)).rx2("p.value")[0]
    return pval


def dhsic(X, Y, kernel=["gaussian", "gaussian"]):
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    res = dHSIC.dhsic(X=X, Y=Y, kernel=kernel)

    return res


def fit_restart(trainloader, hsic_net, pval_callback, max_epochs, se_callback=None, num_restart=10, alpha=0.05):
    max_pval = 0.0
    best_model = None
    for i in range(num_restart):
        # restart params
        hsic_net.initialize()
        trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[se_callback, pval_callback])
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
