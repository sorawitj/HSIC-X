import sys

sys.path.extend(['DeepGMM'])

from DeepGMM.methods.toy_model_selection_method import ToyModelSelectionMethod
from kernel import CategoryKernel, RBFKernel
from model import train_HSIC_IV, NonlinearModel
import pandas as pd
from utils import med_sigma, to_torch, gen_data, fit_restart, gen_radial_fn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(1)

vis_mode = False
compare_mode = True
n_rep = 10
n = 1000
num_basis = 10
data_limits = (-7, 7)
max_epoch_hsic = {'linear': 300, 'radial': 400}
num_restart = 4

f = gen_radial_fn(num_basis=num_basis, data_limits=data_limits)

X, Y, Z, x_vis = gen_data(f, n, 'mix_Gaussian', alpha=1)
X_dev, Y_dev, Z_dev, _ = gen_data(f, n, 'mix_Gaussian', alpha=1)
dat = [X, Z, Y, X_dev, Z_dev, Y_dev]
# to torch
for i in range(len(dat)):
    dat[i] = to_torch(dat[i]).double()

method = ToyModelSelectionMethod()
method.fit(*dat, g_dev=None, verbose=True)

g_test = method.predict(to_torch(x_vis).double())
y_deepGMM = g_test.flatten().detach().numpy()

config = {'batch_size': 256,
          'lr': 5e-2,
          'max_epoch': 1000,
          'num_restart': 1}
kernel_e = RBFKernel(sigma=1)
kernel_z = RBFKernel(sigma=1)
hsic_net = NonlinearModel(input_dim=1,
                          lr=config['lr'],
                          kernel_e=kernel_e,
                          kernel_z=kernel_z)

hsic_net = train_HSIC_IV(hsic_net, config, X, Y, Z)

intercept_adjust = Y.mean() - hsic_net(to_torch(X)).mean()
y_hsic = intercept_adjust + hsic_net(to_torch(x_vis))
y_hsic = y_hsic.detach().numpy()
f_x = f(x_vis)

import matplotlib.pyplot as plt

# X, Y, Z, _ = gen_data(f, 100, 'mix_Gaussian', alpha=1)

plt.scatter(X, Y, c='grey', alpha=.5)
plt.plot(x_vis, y_deepGMM, label='DeepGMM')
plt.plot(x_vis, y_hsic, label='HSIC')
plt.plot(x_vis, f_x, label='f_x')
plt.legend()
plt.xlim(-4, 6)
