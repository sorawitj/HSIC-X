import numpy as np
import torch
from scipy.spatial.distance import pdist


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
    centres = torch.linspace(data_limits[0], data_limits[1], num_basis)

    Phi = torch.zeros((x.shape[0], 2 + num_basis))
    for i in range(num_basis):
        Phi[:, i:i + 1] = torch.exp(-1 * ((x - centres[i]) / 1) ** 2)
    Phi[:, [num_basis]] = x
    Phi[:, [num_basis + 1]] = x ** 2
    return Phi


def gen_radial_fn(num_basis, data_limits=(-5, 5), ret_w=False):
    w = np.random.normal(0, 2, size=num_basis)
    w = np.concatenate([w, [1.5, -0.2]])
    f = lambda x: radial(x, num_basis=num_basis, data_limits=data_limits) @ w
    if ret_w:
        return f, w
    else:
        return f


def sample_cov_matrix(d, k, seed, ind=False, var_diff=True):
    rand_state = np.random.RandomState(seed)
    if ind:
        cov = np.eye(d)
    else:
        W = rand_state.randn(d, k)
        S = W.dot(W.T) + np.diag(rand_state.rand(d))
        inv_std = np.diag(1. / np.sqrt(np.diag(S)))
        cov = inv_std @ S @ inv_std

    if var_diff:
        var = rand_state.uniform(0.5, 3, size=(d,))
        std = np.diag(np.sqrt(var))
        cov = std @ cov @ std

    return cov


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
    U = rnorm(n)
    e_y = rnorm(n)
    e_x = rnorm_d(n, x_dim)

    bZ1 = np.random.RandomState(5).uniform(-4, 4, size=(z_dim, x_dim))
    bZ2 = .5 * np.ones(shape=(z_dim, x_dim))

    if oracle:
        g = 0
    else:
        g = 1

    if iv_type == 'mix_Gaussian':
        Z = rnorm_d(n, z_dim)
    elif iv_type == 'mix_Binary':
        Z = (-2) ** np.random.binomial(1, p=np.repeat(1 / 3, z_dim), size=(n, z_dim))
    elif iv_type == 'mix_CorGaussian':
        cov = sample_cov_matrix(z_dim, z_dim, seed=0,
                                ind=True, var_diff=False)
        Z = rnorm_d(n, z_dim, mean=np.zeros(z_dim), cov=cov)
    else:
        raise Exception('Invalid iv_type {}'.format(iv_type))

    X = e_x * (Z ** 2 @ bZ2) + g * U[:, np.newaxis] + alpha * Z @ bZ1
    Y = f(X) - 2 * U + e_y  # linear

    return X, Y, Z


def med_sigma(x):
    if x.ndim == 1:
        x = x[:, np.newaxis]

    return np.sqrt(np.median(pdist(x, 'sqeuclidean')) * .5)


def get_med_sigma(hsic_net, data, s_z=False):
    X, Y, Z = data
    res = (Y - hsic_net(X)).detach().numpy()
    if s_z:
        return med_sigma(res), med_sigma(Z)
    else:
        return med_sigma(res)


def rnorm(n):
    return np.random.normal(size=n)


def rnorm_d(n, d, mean=None, cov=None):
    if mean is None:
        return np.random.normal(size=(n, d))
    else:
        return np.random.multivariate_normal(mean, cov, size=n)


def to_torch(arr):
    return torch.from_numpy(np.array(arr).astype(np.float32))
