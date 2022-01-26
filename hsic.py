import torch
import numpy as np

from utils import to_torch


def HSIC(x, y, kernel_x, kernel_y, K=None, L=None):
    if x.dim() == 1:
        x = x.view(x.size(0), -1)
    if y.dim() == 1:
        y = y.view(y.size(0), -1)

    m, _ = x.shape  # batch size

    if K is None:
        K = kernel_x(x)
    if L is None:
        L = kernel_y(y)

    H = torch.eye(m, dtype=torch.float32) - 1.0 / m * torch.ones((m, m), dtype=torch.float32)

    HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
    return HSIC


def hsic_perm_test(X, Y, kernel_x, kernel_y, B=100):
    X, Y = to_torch(X).detach(), to_torch(Y).detach()
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

    p_value = (np.sum(hsic_perm >= hsic) + 1) / (B + 1)

    return hsic, p_value
