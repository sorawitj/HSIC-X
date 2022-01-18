import torch



def pairwise_distances(x):
    # x should be two dimensional
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ / (2 * sigma ** 2))


def HSIC(x, y, kernel_x, kernel_y):
    if x.dim() == 1:
        x = x.view(x.size(0), -1)
    if y.dim() == 1:
        y = y.view(y.size(0), -1)

    m, _ = x.shape  # batch size

    K = kernel_x(x)
    L = kernel_y(y)

    H = torch.eye(m, dtype=torch.float32) - 1.0 / m * torch.ones((m, m), dtype=torch.float32)

    HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
    return HSIC
