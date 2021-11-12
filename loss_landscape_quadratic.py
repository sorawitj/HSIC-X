import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from torch.utils.data import TensorDataset
import seaborn as sns

from kernel import PolyKernel, RBFKernel
from model import LinearModel, NonlinearModel, PolyModel
from utils import *
import pytorch_lightning as pl
import matplotlib.pyplot as plt

n = 10000
# f = lambda x: 2*(1 / (1 + np.exp(-2 * x)))
# f = lambda x: np.log(np.abs(16. * x - 0) + 1) * np.sign(x - 0)
f = lambda x: 3 * x
# f = lambda x: 2 * x + .5 * x ** 2

X, Y, Z, U, e_y, _ = gen_data(f, n, iv_type='diffcoef')
np.mean(Z ** 2 * U)
np.mean(Z * U)
np.mean(U[Z == 1]) * np.mean(Z == 1) - 2 * np.mean(U[Z == -2]) * np.mean(Z == -2)
np.mean(U[Z == 1]) * np.mean(Z == 1) + 4 * np.mean(U[Z == -2]) * np.mean(Z == -2)

(1 / np.mean(X * Z)) * np.mean(Z * Y)

np.mean(Z ** 2 * U)
np.mean(X * Z)

np.mean(Z * Y)
3 * np.mean(Z ** 2 * U) - 4 * np.mean(Z * U) + np.mean(Z * e_y)

np.mean(Z * Y)

batch_size = 128  # 256
trainloader = torch.utils.data.DataLoader(TensorDataset(to_torch(X), to_torch(Y), to_torch(Z)),
                                          batch_size=batch_size,
                                          shuffle=True, num_workers=0)

# for quadratic functon
s_z = med_sigma(Z)
kernels = PolyKernel(degree=2, s_z=0.1)
hsic_net = PolyModel(1, lr=1e-2, lmd=-99, kernels=kernels, bias=False)

df_loss = pd.DataFrame(columns=['loss', 'param'])
data = (to_torch(X), to_torch(Y), to_torch(Z))
param_grid, loss, grad_u, grad_v = get_loss_landscape_poly(hsic_net, data, 40)

np.save('results/param_grid_quadratic.npy', param_grid)
np.save('results/loss_quadratic.npy', loss)
np.save('results/grad_u_quadratic.npy', grad_u)
np.save('results/grad_v_quadratic.npy', grad_v)
np.save('results/X.npy', X)
np.save('results/Y.npy', Y)
np.save('results/Z.npy', Z)

param_grid = np.load('results/param_grid_quadratic.npy')
loss = np.load('results/loss_quadratic.npy')
grad_u = np.load('results/grad_u_quadratic.npy')
grad_v = np.load('results/grad_v_quadratic.npy')
# plot different initialization
xy_min = [-10, -5]
xy_max = [10, 5]
good_res = []
bad_res = []
for i in range(10):
    weights = np.random.uniform(low=xy_min, high=xy_max, size=(1, 2))
    # hsic_net.layers[0].bias.data = torch.Tensor([0.])
    hsic_net.layers[0].weight.data = torch.Tensor(weights)
    trainer = pl.Trainer(max_epochs=1000)
    trainer.fit(hsic_net, trainloader)
    res = hsic_net.layers[0].weight.data.detach().numpy().flatten()
    print(weights)
    print(res)
    if abs(2 - res[0]) < 0.8 and abs(0.5 - res[1]) < 0.2:
        good_res += [weights.flatten()]
    else:
        bad_res += [weights.flatten()]

xvals, yvals = param_grid
zvals = loss
contours = plt.contour(xvals, yvals, zvals, colors='black', linestyles='dashed')
plt.clabel(contours, inline=True, fontsize=8)
cp = plt.contourf(xvals, yvals, zvals, cmap='viridis_r')
# grads
norm = np.linalg.norm(np.array((grad_u, grad_v)), axis=0)
grad_u = -grad_u / norm
grad_v = -grad_v / norm

contours = plt.contour(xvals, yvals, zvals, levels=12, colors='black', linestyles='dashed')
plt.clabel(contours, inline=True, fontsize=8)
plt.contourf(xvals, yvals, zvals, cmap='viridis_r')
plt.quiver(xvals, yvals, grad_v, grad_u, units='xy',
           scale=3, scale_units='xy', color='gray',
           width=0.055, headwidth=3, alpha=0.8)
plt.plot([.5], [2], 'rx', markersize=4)
plt.plot([j for i, j in good_res], [i for i, j in good_res], 'go', markersize=4)
plt.plot([j for i, j in bad_res], [i for i, j in bad_res], 'ro', markersize=4)
plt.title("Model: Quadratic, Instrument: Var")
plt.xlabel('X**2')
plt.ylabel('X')
# plt.savefig('loss_landscape_quadratic_diffcoef.pdf')
