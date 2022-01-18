from scipy.spatial.distance import pdist
from torch.utils.data import TensorDataset
from hsic import *
from kernel import CategoryKernel, PTKGauss
from model import LinearModel, MedianHeuristic, Poly2SLS, PvalueLog, PolyModel, NonlinearModel, RadialModel, Radial2SLS, \
    Radial2SLSRidge, PredPolyRidge, PolySLSRidge
import pandas as pd
from utils import med_sigma, to_torch, gen_data, fit_restart, gen_radial_fn, gen_data_multi, dhsic, dhsic_test
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(1)
compare_mode = True
n_rep = 10
n = 1000
max_epoch_hsic = {'linear': 300, 'radial': 400}
# max_epoch_mse = {'linear': 100, 'radial': 300}
num_restart = 1

np.random.seed(1)
x_dim = 1
z_dim = 5
# u_dim = 2
# w = np.random.normal(0, 2, size=(x_dim, 1))
w = np.array([[-2]])
# f = lambda x: (x @ w).flatten()
f = lambda x: (x @ w).flatten()

fn = 'linear'
for instrument in ['Gaussian']:
    # for instrument in ['Gaussian', 'Binary']:
    res_df = pd.DataFrame(columns=['f_x', 'Pred', 'IND', '2SLS', 'Oracle', 'alpha', 'z_dim', 'run_id'])

    # get a fix x_vis
    iv_type = 'mix_{}'.format(instrument)
    # iv_type = 'mean'
    # alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    alphas = np.linspace(0, 1, 5)
    # lr_s = np.linspace(5e-2, 2e-1, 5)
    # lr_s = [5e-2, 7e-2, 9e-2, 1e-1, 2e-1]
    lr_s = [5e-2, 5e-2, 5e-2, 5e-2, 5e-2]
    alphas = [.4]
    lr_s = [5e-2]
    for z_dim in [1, 2, 3, 4, 5]:
        for j in range(len(alphas)):
            alpha = alphas[j]
            lr = lr_s[j]
            for i in range(n_rep):
                X, Y, Z = gen_data_multi(f, n, x_dim, z_dim, iv_type, alpha=alpha, oracle=False)
                X_o, Y_o, _ = gen_data_multi(f, n, x_dim, z_dim, iv_type, alpha=alpha, oracle=True)
                X_test, _, _ = gen_data_multi(f, int(10e4), x_dim, z_dim, iv_type, alpha=alpha, oracle=False)

                batch_size = 128  # 256
                trainloader = torch.utils.data.DataLoader(TensorDataset(to_torch(X), to_torch(Y), to_torch(Z)),
                                                          batch_size=batch_size,
                                                          shuffle=True, num_workers=0)

                # get y_hat for MSE loss
                mse_reg = PredPolyRidge(degree=1, bias=False)
                oracle_reg = PredPolyRidge(degree=1, bias=False)
                mse_reg.fit(X, Y)
                y_hat_mse = mse_reg.predict(X_test)
                mse_coef = mse_reg.reg.coef_
                oracle_reg.fit(X_o, Y_o)
                y_hat_oracle = oracle_reg.predict(X_test)

                # s_z = med_sigma(Z)
                s_z = med_sigma(Z)

                # kernels = PolyKernel(degree=1)
                # kernel_e = RBFKernel(sigma=1)
                kernel_e = PTKGauss(sigma=1)
                if instrument == 'Binary':
                    kernel_z = CategoryKernel()
                else:
                    # kernel_z = RBFKernel(sigma=s_z)
                    kernel_z = PTKGauss(sigma=s_z)
                # hsic_net = NonlinearModel(1, 5e3, 1e-2, kernel_e, kernel_z)
                if fn == 'linear':
                    hsic_net = LinearModel(input_dim=x_dim,
                                           lr=lr,
                                           lmd=0.0,
                                           gamma=0.0,  # 0.05
                                           kernel_e=kernel_e,
                                           kernel_z=kernel_z,
                                           bias=False)
                else:
                    hsic_net = RadialModel(input_dim=1,
                                           num_basis=num_basis,
                                           data_limits=data_limits,
                                           lr=lr,
                                           lmd=0.003,
                                           gamma=0.0,  # 0.05
                                           kernel_e=kernel_e,
                                           kernel_z=kernel_z,
                                           bias=False)

                # hsic_net.load_state_dict(mse_coef)

                se_callback = MedianHeuristic()
                pval_callback = PvalueLog(B=50)

                hsic_net = fit_restart(trainloader, hsic_net, pval_callback, max_epoch_hsic[fn], se_callback,
                                       num_restart=num_restart)

                # y_hat = hsic_net(to_torch(X))
                #
                # HSIC(to_torch(Y) - y_hat, to_torch(Z), hsic_net.kernels.e, hsic_net.kernels.z)
                # HSIC(to_torch(Y) - to_torch(f(X)), to_torch(Z), hsic_net.kernels.e, hsic_net.kernels.z)
                # HSIC(to_torch(Y) - to_torch(mse_reg.predict(X)), to_torch(Z), hsic_net.kernels.e, hsic_net.kernels.z)
                # HSIC(to_torch(Y) - to_torch(oracle_reg.predict(X)), to_torch(Z), hsic_net.kernels.e, hsic_net.kernels.z)
                #
                # s_e, s_z = dhsic(Y - y_hat.detach().numpy(), Z).rx2("bandwidth")
                # k_e = PTKGauss(s_e)
                # k_z = PTKGauss(s_z)
                # HSIC(to_torch(Y) - y_hat, to_torch(Z), k_e, k_z)
                # HSIC(to_torch(Y) - to_torch(f(X)), to_torch(Z), k_e, k_z)
                # # np.sqrt(np.median(pdist(Z, 'sqeuclidean')) * .5)
                # dhsic(Y - y_hat.detach().numpy(), Z).rx2("dHSIC")
                # dhsic(Y - f(X), Z).rx2("dHSIC")
                # dhsic_test(Y - y_hat.detach().numpy(), Z)

                intercept_adjust = Y.mean() - hsic_net(to_torch(X)).mean()
                y_hat_hsic = intercept_adjust + hsic_net(to_torch(X_test))

                # 2SLS
                # poly2SLS = Radial2SLS(num_basis=num_basis, data_limits=data_limits, bias=False)
                if fn == 'linear':
                    poly2SLS = Poly2SLS(degree=1, bias=False)
                    # poly2SLS = PolySLSRidge(degree=1, bias=False)
                else:
                    poly2SLS = Radial2SLSRidge(num_basis=num_basis, data_limits=data_limits, bias=False)

                poly2SLS.fit(X, Y, Z)

                inner_df = pd.DataFrame(columns=res_df.columns)

                inner_df['f_x'] = f(X_test)
                inner_df['Pred'] = y_hat_mse
                inner_df['IND'] = y_hat_hsic.detach().numpy()
                inner_df['2SLS'] = poly2SLS.predict(X_test)
                inner_df['Oracle'] = y_hat_oracle
                inner_df['alpha'] = alpha
                inner_df['z_dim'] = z_dim
                inner_df['run_id'] = i

                res_df = res_df.append(inner_df, ignore_index=True)

    if compare_mode:
        melt_res_df = res_df.melt(id_vars=['f_x', 'z_dim', 'alpha', 'run_id'], var_name='model',
                                  value_name='y_pred')
        melt_res_df['Mean Square Error'] = (melt_res_df['f_x'] - melt_res_df['y_pred']) ** 2
        final_df = melt_res_df.groupby(['model', 'z_dim', 'alpha', 'run_id'])['Mean Square Error'].mean().reset_index()
        final_df['alpha'] = np.round(final_df.alpha, 2)
        final_df.to_csv("results/compare_df_multidim_fn_{}_ins_{}.csv".format(fn, instrument),
                        index=False)
        final_df = pd.read_csv("results/compare_df_multidim_fn_{}_ins_{}.csv".format(fn, instrument))
        # g = sns.boxplot(data=final_df, x='alpha', y='Mean Square Error', hue='model')
        g = sns.catplot(data=final_df, kind="point", log=True,
                        x='z_dim', y='Mean Square Error', hue='model',
                        markers=["o", "x", "d", "s"], linestyles=[':', '--', '-', '-.'])
        g.fig.get_axes()[0].set_yscale('log')

        plt.title("Model: {}, Instrument: {}".format(fn, instrument))
        # # plt.ylim(1e-2, 1e4)
        # plt.tight_layout()
        # plt.savefig('results/compare_df_multidim_fn__{}_ins_{}.pdf'.format(fn, instrument))
        # plt.close()
