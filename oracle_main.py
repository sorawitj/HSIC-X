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
var_effect = True
data_limits = (-7, 7)
max_epoch_mse = {'linear': 100, 'radial': 200}

for fn in ['radial', 'linear']:
    if fn == 'linear':
        f = lambda x: -2 * x
    else:
        f = gen_radial_fn(num_basis=num_basis, data_limits=data_limits)
    for instrument in ['Gaussian', 'Binary']:

        res_df = pd.DataFrame(columns=['x_vis', 'f_x', 'Oracle', 'alpha', 'run_id'])

        # get a fix x_vis
        iv_type = 'mix_{}'.format(instrument)
        # iv_type = 'mean'
        _, _, _, X_vis = gen_data(f, n, iv_type, var_effect=var_effect, gamma=1)
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
                X, Y, Z, _ = gen_data(f, n, iv_type, alpha=alpha, var_effect=var_effect, gamma=0)
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

                inner_df = pd.DataFrame(columns=res_df.columns)

                inner_df['x_vis'] = X_test
                inner_df['f_x'] = f(X_test)
                inner_df['Oracle'] = y_hat_mse.detach().numpy()
                inner_df['alpha'] = alpha
                inner_df['run_id'] = i

                res_df = res_df.append(inner_df, ignore_index=True)

        if compare_mode:
            melt_res_df = res_df.melt(id_vars=['x_vis', 'f_x', 'alpha', 'run_id'], var_name='model',
                                      value_name='y_pred')
            melt_res_df['Mean Square Error'] = (melt_res_df['f_x'] - melt_res_df['y_pred']) ** 2
            final_df = melt_res_df.groupby(['model', 'alpha', 'run_id'])['Mean Square Error'].mean().reset_index()
            final_df['alpha'] = np.round(final_df.alpha, 2)
            final_df.to_csv("results/compare_df_oracle_fn_{}_ins_{}_var_{}.csv".format(fn, instrument, var_effect),
                            index=False)

# Plotting

for fn in ['linear', 'radial']:
    for instrument in ['Binary', 'Gaussian']:
        final_df = pd.read_csv("results/compare_df_fn_{}_ins_{}.csv".format(fn, instrument))
        final_df = final_df.replace('y_2sls', '2SLS')
        final_df = final_df.replace('y_mse', 'Pred')
        final_df = final_df.replace('y_hsic', 'IND')
        oracle_df = pd.read_csv("results/compare_df_oracle_fn_{}_ins_{}_var_{}.csv".format(fn, instrument, var_effect))
        final_df = final_df.append(oracle_df, ignore_index=True)
        final_df = final_df.rename({'model': 'Method'}, axis=1)
        # g = sns.boxplot(data=final_df, x='alpha', y='Mean Square Error', hue='model')
        g = sns.catplot(data=final_df, kind="point", log=True,
                        x='alpha', y='Mean Square Error', hue='Method',
                        markers=["o", "x", "d", "s"], linestyles=[':', '--', '-', '-.'])
        g.fig.get_axes()[0].set_yscale('log')

        plt.title("")
        # plt.ylim(1e-2, 1e4)
        # plt.tight_layout()
        plt.savefig('results/compare_alpha_fn_{}_ins_{}_var_{}.pdf'.format(fn, instrument, var_effect),
                    bbox_inches="tight")
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
