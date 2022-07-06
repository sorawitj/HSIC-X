import numpy as np
import pytorch_lightning as pl

from helpers.hsic import dhsic_test
from helpers.utils import get_med_sigma, med_sigma
from models.kernel import RBFKernel, CategoryKernel


class MedianHeuristic(pl.Callback):
# a callback to update the kernel parameter with median heuristic after each epoch ends

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

    def on_train_epoch_end(self, trainer, pl_module):
        if self.epoch < 1000:
            data = trainer.train_dataloader.dataset.datasets.tensors
            s_e = get_med_sigma(pl_module, data)
            if pl_module.kernels.e.__class__ == RBFKernel:
                pl_module.kernels.e.set_kernel_param(s_e)
        self.epoch += 1


class PvalueLog(pl.Callback):
# a callback to compute p-value of the HSIC independence test after the training has ended

    def __init__(self, every_epoch=False, B=100):
        self.every_epoch = every_epoch
        self.B = B
        self.p_value = None

    def print_pval(self, trainer, pl_module):
        X, Y, Z = trainer.train_dataloader.dataset.datasets.tensors
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


class PvalueOptim(pl.Callback):

    def on_train_epoch_end(self, trainer, pl_module):
        data = trainer.train_dataloader.dataset.datasets.tensors
        best_s_e, min_pvalue = optimize_pval(data, pl_module)
        pl_module.s_e = best_s_e
        print("min p-value: {}, update s_e: {}".format(min_pvalue, best_s_e))
