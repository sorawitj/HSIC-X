from torch.utils.data import TensorDataset
from hsic import *
from kernel import PolyKernel, RBFKernel, CategoryKernel
from model import LinearModel, MedianHeuristic, Poly2SLS, PvalueLog, PolyModel, NonlinearModel, RadialModel, Radial2SLS, \
    Radial2SLSRidge
import pytorch_lightning as pl
import pandas as pd
from utils import med_sigma, to_torch, gen_data, fit_restart, gen_radial_fn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(1)
vis_mode = True
compare_mode = True
n_rep = 10
n = 1000
num_basis = 10
var_effect = False
data_limits = (-7, 7)
max_epoch_hsic = {'linear': 100, 'radial': 600}
max_epoch_mse = {'linear': 100, 'radial': 300}
num_restart = 5
# f = lambda x: 2*(1 / (1 + np.exp(-2 * x)))
# f = lambda x: np.log(np.abs(16. * x - 0) + 1) * np.sign(x - 0)
# f = lambda x: 3 * x
# f = lambda x: 2 * x + .5 * x ** 2

for fn in ['radial', 'linear']:
    if fn == 'linear':
        f = lambda x: -2 * x
    else:
        f = gen_radial_fn(num_basis=num_basis, data_limits=data_limits)
    for instrument in ['Gaussian', 'Binary']:

        res_df = pd.DataFrame(columns=['x_vis', 'f_x', 'y_mse', 'y_hsic', 'alpha', 'run_id'])
        res_df_vis = pd.DataFrame(columns=['x_vis', 'f_x', 'y_mse', 'y_hsic', 'alpha', 'run_id'])

        # get a fix x_vis
        iv_type = 'mix_{}'.format(instrument)
        # iv_type = 'mean'
        _, _, _, X_vis = gen_data(f, n, iv_type, var_effect=var_effect)
        # alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        if fn == 'linear':
            alphas = [0, 0.1, 0.2, 0.3, 0.4]
        else:
            alphas = np.linspace(0, 1, 5)
        # lr_s = np.linspace(5e-2, 2e-1, 5)
        if fn == 'linear':
            lr = 1e-2
            lr_s = np.repeat(lr, 5)
        else:
            lr_s = [5e-2, 7e-2, 9e-2, 1e-1, 2e-1]
        # alphas = [0]
        for j in range(len(alphas)):
            alpha = alphas[j]
            lr = lr_s[j]
            for i in range(n_rep):
                X, Y, Z, _ = gen_data(f, n, iv_type, alpha=alpha, var_effect=var_effect)
                X_test, _, _, _ = gen_data(f, X_vis.shape[0], iv_type, alpha=alpha, var_effect=var_effect)

                batch_size = 256  # 256
                trainloader = torch.utils.data.DataLoader(TensorDataset(to_torch(X), to_torch(Y), to_torch(Z)),
                                                          batch_size=batch_size,
                                                          shuffle=True, num_workers=0)

                # mse_net = PolyModel(input_dim=1, degree=1, lr=1e-1, bias=False)
                if fn == 'linear':
                    mse_net = LinearModel(input_dim=1, lr=1e-1)
                else:
                    mse_net = RadialModel(input_dim=1, num_basis=num_basis, data_limits=data_limits,
                                          lr=1e-1, bias=False)
                # mse_net = NonlinearModel(1, lr=1e-2)
                # mse_net = LinearModel(1, lr=1e-1)
                trainer = pl.Trainer(max_epochs=max_epoch_mse[fn])
                trainer.fit(mse_net, trainloader)
                # get y_hat for MSE loss
                y_hat_mse = mse_net(to_torch(X_test))
                y_hat_mse_vis = mse_net(to_torch(X_vis))

                s_z = med_sigma(Z)
                # kernels = PolyKernel(degree=1)
                kernel_e = RBFKernel(sigma=1)
                if instrument == 'Binary':
                    kernel_z = CategoryKernel()
                else:
                    kernel_z = RBFKernel(sigma=s_z)
                # hsic_net = NonlinearModel(1, 5e3, 1e-2, kernel_e, kernel_z)
                if fn == 'linear':
                    hsic_net = LinearModel(input_dim=1,
                                           lr=lr,
                                           lmd=-99,
                                           gamma=0.0,  # 0.05
                                           kernel_e=kernel_e,
                                           kernel_z=kernel_z,
                                           bias=False)
                else:
                    hsic_net = RadialModel(input_dim=1,
                                           num_basis=num_basis,
                                           data_limits=data_limits,
                                           lr=lr,
                                           lmd=-99,
                                           gamma=0.0,  # 0.05
                                           kernel_e=kernel_e,
                                           kernel_z=kernel_z,
                                           bias=False)
                if fn == 'radial':
                    hsic_net.load_state_dict(mse_net)

                se_callback = MedianHeuristic()
                pval_callback = PvalueLog(B=50)

                hsic_net = fit_restart(trainloader, hsic_net, pval_callback, max_epoch_hsic[fn], se_callback,
                                       num_restart=num_restart)

                intercept_adjust = Y.mean() - hsic_net(to_torch(X)).mean()
                y_hat_hsic = intercept_adjust + hsic_net(to_torch(X_test))
                y_hat_hsic_vis = intercept_adjust + hsic_net(to_torch(X_vis))

                # 2SLS
                # poly2SLS = Radial2SLS(num_basis=num_basis, data_limits=data_limits, bias=False)
                if fn == 'linear':
                    poly2SLS = Poly2SLS(degree=1, bias=False)
                else:
                    poly2SLS = Radial2SLSRidge(num_basis=num_basis, data_limits=data_limits, bias=False)

                poly2SLS.fit(X, Y, Z)

                inner_df = pd.DataFrame(columns=res_df.columns)
                inner_df_vis = pd.DataFrame(columns=res_df.columns)

                inner_df['x_vis'] = X_test
                inner_df['f_x'] = f(X_test)
                inner_df['y_mse'] = y_hat_mse.detach().numpy()
                inner_df['y_hsic'] = y_hat_hsic.detach().numpy()
                inner_df['y_2sls'] = poly2SLS.predict(X_test)
                inner_df['alpha'] = alpha
                inner_df['run_id'] = i

                inner_df_vis['x_vis'] = X_vis
                inner_df_vis['f_x'] = f(X_vis)
                inner_df_vis['y_mse'] = y_hat_mse_vis.detach().numpy()
                inner_df_vis['y_hsic'] = y_hat_hsic_vis.detach().numpy()
                inner_df_vis['y_2sls'] = poly2SLS.predict(X_vis)
                inner_df_vis['alpha'] = alpha
                inner_df_vis['run_id'] = i

                res_df = res_df.append(inner_df, ignore_index=True)
                res_df_vis = res_df_vis.append(inner_df_vis, ignore_index=True)

        if compare_mode:
            melt_res_df = res_df.melt(id_vars=['x_vis', 'f_x', 'alpha', 'run_id'], var_name='model',
                                      value_name='y_pred')
            melt_res_df['Mean Square Error'] = (melt_res_df['f_x'] - melt_res_df['y_pred']) ** 2
            final_df = melt_res_df.groupby(['model', 'alpha', 'run_id'])['Mean Square Error'].mean().reset_index()
            final_df['alpha'] = np.round(final_df.alpha, 2)
            final_df.to_csv("results/compare_df_fn_{}_ins_{}_var_{}.csv".format(fn, instrument, var_effect),
                            index=False)
            # g = sns.boxplot(data=final_df, x='alpha', y='Mean Square Error', hue='model')
            g = sns.catplot(data=final_df, kind="point", log=True,
                            x='alpha', y='Mean Square Error', hue='model',
                            markers=["o", "x", "d"], linestyles=[':', '--', '-'])
            g.fig.get_axes()[0].set_yscale('log')

            plt.title("Model: {}, Instrument: {}".format(fn, instrument))
            # plt.ylim(1e-2, 1e4)
            plt.tight_layout()
            plt.savefig('results/compare_alpha_fn_{}_ins_{}_var_{}.pdf'.format(fn, instrument, var_effect))
            plt.close()

        if vis_mode:
            res_df_vis.to_csv("results/vis_df_fn_{}_ins_{}_var_{}.csv".format(fn, instrument, var_effect), index=False)
            for alpha in alphas:
                res_alpha = res_df_vis[res_df_vis.alpha == alpha]
                melt_res_df = res_alpha.melt(id_vars=['x_vis', 'alpha', 'run_id'], var_name='model', value_name='y')
                X, Y, _, _ = gen_data(f, n, iv_type, alpha=alpha, var_effect=var_effect)
                sns.scatterplot(X, Y, color='.5', linewidth=0, alpha=0.5)
                sns.lineplot(data=melt_res_df.query("model!='f_x'"), x="x_vis", y="y", units='run_id',
                             hue='model', estimator=None, lw=1, alpha=0.5,
                             palette=["blue", "red", "green"])
                df_0 = res_alpha.query("run_id == 0.0")
                plt.plot(df_0['x_vis'].values, df_0['f_x'].values, '--',
                         label='f_x', c='black', lw=1.8)
                plt.legend()
                plt.ylim(Y.min() * 1.2, Y.max() * 1.2)
                # plt.xlim(-8, 8)
                plt.title("Model: {}, Instrument: {}, Alpha: {}".format(fn, instrument, (alpha)))
                plt.savefig(
                    'results/vis_plot_radial_fn_{}_ins_{}_alpha_{}_var_{}.pdf'.format(fn, instrument, str(alpha),
                                                                                      var_effect))
                plt.close()

# #
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
#
# linReg = LinearRegression(fit_intercept=False)
# poly = PolynomialFeatures(degree=2, include_bias=False)
# #
# X_poly = poly.fit_transform(X[:, np.newaxis])
# # centering
# Z_poly = poly.fit_transform(Z[:, np.newaxis])
#
# X_poly = X_poly - np.mean(X_poly, axis=0)
# Z_poly = Z_poly - np.mean(Z_poly, axis=0)
#
# np.cov(X_poly.T, Z_poly.T)
#
# Xhat_poly = poly2SLS.reg1.predict(Z_poly)
# Xhat_poly = Xhat_poly - np.mean(Xhat_poly, axis=0)
#
# reg1 = linReg.fit(Xhat_poly[:, [0]], Y).coef_
# reg2 = linReg.fit(Xhat_poly, Y).coef_
#
# np.linalg.svd(Xhat_poly - np.mean(Xhat_poly, axis=0))[1]
#
# np.sqrt((reg1**2).sum())
# np.sqrt((reg2**2).sum())
#
# np.linalg.det(np.cov(Xhat_poly.T))
#
# np.linalg.det(np.cov(X_poly.T, Z_poly.T))
#
# np.linalg.inv(Xhat_poly.T@Z_poly)@Z_poly.T@Y
#
# #
# np.linalg.inv(Z_poly.T@(X_poly))@Z_poly.T@Y
# # poly2SLS.reg2.coef_
# #
# # np.linalg.lstsq(Z_poly.T@(X_poly), Z_poly.T@Y)
# #
# # linReg.fit(X_poly, Y)
# # linReg.coef_
# #
# # Z1, Z2 = Z == 1, Z == -2
# #
# # 0.5**2 * (1 + 2 * -2)
# # np.var(X[Z1]) + np.mean(X[Z1])**2
# # np.mean(X[Z1]**2)
# #
# # np.mean(X[Z2]**2)
