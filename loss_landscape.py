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

n = 1000
# f = lambda x: 2*(1 / (1 + np.exp(-2 * x)))
# f = lambda x: np.log(np.abs(16. * x - 0) + 1) * np.sign(x - 0)
f = lambda x: 3 * x
# f = lambda x: 2 * x + .5 * x ** 2


plot1D = False
plot2D = True

X, Y, Z, _ = gen_data(f, n, iv_type='var')

batch_size = 256  # 256
trainloader = torch.utils.data.DataLoader(TensorDataset(to_torch(X), to_torch(Y), to_torch(Z)),
                                          batch_size=batch_size,
                                          shuffle=True, num_workers=0)

# # for quadratic functon
# s_z = med_sigma(Z)
# kernels = PolyKernel()
# hsic_net = PolyModel(1, lr=1e-2, lmd=-99, kernels=kernels)
# mse_net = PolyModel(1, lr=1e-2, kernels=kernels)
# trainer = pl.Trainer(max_epochs=200)
# trainer.fit(mse_net, trainloader)
# df_loss = pd.DataFrame(columns=['loss', 'param'])
# data = (to_torch(X), to_torch(Y), to_torch(Z))
# param_grid, loss = get_loss_landscape_poly(hsic_net, data)
# np.save('param_grid_quadratic.npy', param_grid)
# np.save('loss_quadratic.npy.npy', loss)
#
# xvals, yvals = param_grid
# zvals = np.log(loss)
# contours = plt.contour(yvals, xvals, zvals, levels=10, colors='black', linestyles='dashed')
# plt.clabel(contours, inline=True, fontsize=8)
# cp = plt.contourf(yvals, xvals, zvals, cmap='viridis_r')
# plt.plot([2.], [0.5], 'ro', markersize=4)
# plt.title("Model: Quadratic, Instrument: DiffCoef")
# plt.xlabel('X')
# plt.ylabel('X**2')
# plt.ylim(-7, 7)
# plt.xlim(-5, 17)
# plt.savefig('loss_landscape_quadratic_diffcoef.pdf')
#
# sns.lineplot(x='param', y='loss', data=df_loss, legend='full')
# plt.axvline(3, 0, 1, c='red', linestyle='--', label='Causal')
# plt.axvline(mse_net.layers[0].weight.item(), 0, 1, c='blue', linestyle='--', label='OLS')
# plt.legend(title='kernel_scale (1e^k)')
# plt.title("Model: Linear, Instrument: Mean")

s_z = med_sigma(Z)
hsic_net = LinearModel(1, lmd=-99, kernels=RBFKernel())
df_loss = pd.DataFrame(columns=['loss', 'param', 'grad', 's_e', 'med_s_e', 'pval_se'])
data = (to_torch(X), to_torch(Y), to_torch(Z))
se_grid = list(np.logspace(np.log10(1e-3), np.log10(1e5), 15, base=10))

for s_e in se_grid:
    hsic_net.kernels.set_kernel_param(s_e=s_e)

    data = (to_torch(X), to_torch(Y), to_torch(Z))
    params, loss, grad, med_s_e, pval_se = get_loss_landscape(hsic_net, data)
    # params, loss, med_s_e = get_pvalue_landscape(hsic_net, data)
    df_loss = df_loss.append(pd.DataFrame.from_dict(
        {'loss': loss, 'grad': grad, 'med_s_e': med_s_e, 'pval_se': pval_se,
         'param': params.detach().numpy(), 's_e': s_e}), ignore_index=True)

df_loss['kernel_scale'] = np.log10(df_loss.s_e.astype(np.float32)).astype('category')
df_loss['loss_scale'] = df_loss.groupby('kernel_scale')['loss'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
df_loss['med_kernel_scale'] = np.log10(df_loss.med_s_e.astype(np.float32))
df_loss['pval_kernel_scale'] = df_loss.pval_se.apply(lambda x: np.log(x))

if plot1D:
    sns.lineplot(x='param', y='loss_scale', data=df_loss,
                 hue='kernel_scale', legend='full')
    plt.axvline(3, 0, 1, c='red', linestyle='--', label='Causal')
    plt.axvline(mse_net.layers[0].weight.item(), 0, 1, c='blue', linestyle='--', label='OLS')
    plt.legend(title='kernel_scale (1e^k)')
    plt.title("Model: Linear, Instrument: Mean")
    # plt.savefig('loss_landscape_linear_var_pvalue_zoom.pdf')


def extract_pval(x, i):
    if len(x['pval_kernel_scale']) > i:
        return x['param'], x['pval_kernel_scale'][i]
    else:
        return np.nan, np.nan


if plot2D:
    # df_loss = pd.read_csv('df_loss_2.csv')
    # df_loss.to_csv("df_loss_argminpval.csv", index=False)
    df_loss = df_loss.sort_values(by=["param", "kernel_scale"])
    xvals = df_loss["param"].unique()
    yvals = df_loss["kernel_scale"].unique()
    zvals = df_loss["loss_scale"].values.reshape(len(xvals), len(yvals)).T
    med_h = df_loss["med_kernel_scale"].unique()
    pval_h = []
    max_range = df_loss['pval_kernel_scale'].apply(lambda x: len(x)).max()
    for i in range(max_range):
        pval_h += [df_loss.apply(lambda x: extract_pval(x, i), axis=1, result_type='expand').groupby(0).mean().reset_index()]
    # grad
    xgrads = df_loss["grad"].values.reshape(len(xvals), len(yvals)).T
    ygrads = np.zeros_like(xgrads)
    norm = np.linalg.norm(np.array((xgrads, ygrads)), axis=0)
    xgrads = -xgrads / norm

    contours = plt.contour(xvals, yvals, zvals, levels=7, colors='black', linestyles='dashed')
    plt.clabel(contours, inline=True, fontsize=8)
    cp = plt.contourf(xvals, yvals, zvals, cmap='viridis_r')
    plt.quiver(xvals, yvals, xgrads, ygrads, units='xy',
               scale=2, scale_units='xy', angles='xy', color='gray',
               width=0.055, headwidth=3, alpha=0.8)
    plt.plot(xvals, med_h, c='m', marker='.', label='median_heuristic')
    for i in range(max_range):
        plt.plot(pval_h[i][0], pval_h[i][1], c='blue', marker='x', label='pval_heuristic')
    plt.axvline(3, 0, 1, c='red', lw=2, label='Causal')
    plt.legend()
    plt.title("Model: Linear, Instrument: Var")
    plt.xlabel("X")
    plt.ylim(yvals[0], yvals[-1])
    plt.ylabel("Sigma")
    # plt.savefig('loss_landscape_linear_var_contour_grad.pdf')
    # plt.savefig('loss_landscape_linear_var_contour_grad.jpg')

# contours = plt.contour(xvals, yvals, zvals, 30, colors='black')
# plt.clabel(contours, inline=True, fontsize=8)
#
# plt.imshow(zvals, origin='lower',
#            cmap='RdGy', alpha=0.5)
# plt.colorbar();

# df_linear['kernel_scale'] = 'linear'
# sns.lineplot(x='param', y='loss', data=df_linear,
#              hue='kernel_scale', legend='full')
# plt.axvline(3, 0, 1, c='red', linestyle='--', label='Causal')
# plt.axvline(mse_net.layers[0].weight.item(), 0, 1, c='blue', linestyle='--', label='OLS')
# plt.legend(title='kernel')
# plt.title("Model: Linear, Instrument: Mean")
