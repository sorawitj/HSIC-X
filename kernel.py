from pyro.contrib.gp.kernels import RBF, Polynomial
from pyro.nn import PyroModule

from utils import to_torch


class RBFKernel(PyroModule):
    def __init__(self, s_e=1, s_z=1):
        super().__init__()
        self.kernel_e = RBF(1, lengthscale=to_torch(s_e))
        self.kernel_e.lengthscale_unconstrained.requires_grad_(False)
        self.kernel_e.variance_unconstrained.requires_grad_(False)
        self.kernel_z = RBF(1, lengthscale=to_torch(s_z))
        self.kernel_z.lengthscale_unconstrained.requires_grad_(False)
        self.kernel_z.variance_unconstrained.requires_grad_(False)

    def set_kernel_param(self, s_e=None, s_z=None):
        if s_e is not None:
            self.kernel_e.lengthscale = to_torch(s_e)
        if s_z is not None:
            self.kernel_z.lengthscale = to_torch(s_z)


class PolyKernel(PyroModule):
    def __init__(self, degree=2, s_z=None):
        super().__init__()
        self.kernel_e = Polynomial(1, degree=degree, bias=to_torch(0.))
        self.kernel_e.bias_unconstrained.requires_grad_(False)
        self.kernel_e.variance_unconstrained.requires_grad_(False)
        if s_z is None:
            self.kernel_z = Polynomial(1, degree=degree, bias=to_torch(0.))
            self.kernel_z.bias_unconstrained.requires_grad_(False)
            self.kernel_z.variance_unconstrained.requires_grad_(False)
        else:
            self.kernel_z = RBF(1, lengthscale=to_torch(s_z))
            self.kernel_z.lengthscale_unconstrained.requires_grad_(False)
            self.kernel_z.variance_unconstrained.requires_grad_(False)
