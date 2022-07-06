import numpy as np
import torch
from helpers.utils import to_torch
from rpy2.robjects import numpy2ri, packages

numpy2ri.activate()
dHSIC = packages.importr("dHSIC")


def HSIC(x, y, kernel_x, kernel_y, K=None, L=None):
    if x.dim() == 1:
        x = x.view(x.size(0), -1)
    if y.dim() == 1:
        y = y.view(y.size(0), -1)

    m, _ = x.shape

    if K is None:
        K = kernel_x(x)
    if L is None:
        L = kernel_y(y)

    H = torch.eye(m, dtype=torch.float32) - 1.0 / m * torch.ones((m, m), dtype=torch.float32)

    HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
    return HSIC


def hsic_perm_test(X, Y, kernel_x, kernel_y, B=100):
    X, Y = to_torch(X).detach(), to_torch(Y).detach()

    if X.dim() == 1:
        X = X.view(X.size(0), -1)
    if Y.dim() == 1:
        Y = Y.view(Y.size(0), -1)

    # compute kernel matrices
    Kx = kernel_x(X)
    Ky = kernel_y(Y)

    hsic = HSIC(X, Y, kernel_x, kernel_y, K=Kx, L=Ky).detach().item()

    def compute_stat():
        n = Ky.shape[0]
        # permute Y
        idx = np.random.choice(n, n, replace=False)
        new_Ky = Ky[idx, idx[:, None]].clone()
        return HSIC(X, Y, kernel_x, kernel_y, K=Kx, L=new_Ky).detach().item()

    hsic_perm = np.empty(shape=(B,))

    for i in range(B):
        hsic_perm[i] = compute_stat()

    p_value = np.mean(hsic_perm > hsic)

    return p_value, hsic


def dhsic_test(X, Y, kernel=["gaussian", "gaussian"],
               s_x=None, s_y=None, method="gamma", B=10,
               statistics=False):
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if s_x is None:
        res = dHSIC.dhsic_test(X=X, Y=Y,
                               method=method, kernel=kernel,
                               B=B)
    else:
        res = dHSIC.dhsic_test(X=X, Y=Y,
                               B=B, method=method,
                               kernel='gaussian.fixed',
                               bandwidth=(s_x, s_y))
    if statistics:
        ret = res.rx2('p.value')[0], res.rx2('statistic')[0]
    else:
        ret = res.rx2('p.value')[0]
    return ret


def dhsic(X, Y, kernel=["gaussian", "gaussian"]):
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    res = dHSIC.dhsic(X=X, Y=Y, kernel=kernel)

    return res
