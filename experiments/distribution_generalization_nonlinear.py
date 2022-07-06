import functools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
import rpy2.robjects.packages as packages

from helpers.trainer import train_mse, train_HSIC_IV
from models.baselines import PredPolyRidge
from models.hsicx import NNHSICX, LinearHSICX
from models.kernel import RBFKernel

pandas2ri.activate()
numpy2ri.activate()

AR = packages.importr("AnchorRegression")


# define anchor regression function
def train_AR(X, Y, Z):
    df_X = pd.DataFrame(X, columns=['X' + str(i) for i in range(X.shape[1])])
    df_X['Y'] = Y
    df_Z = pd.DataFrame(Z, columns=['Z'])

    ret = AR.anchor_regression(df_X, df_Z, 5e2, 'Y', 0).rx2('coeff')
    return ret


def generate_data(n, int_par, f, oracle=False):
    rand = np.random
    indicator = rand.binomial(1, int_par / 4, size=n)
    Z1 = rand.uniform(0, int_par, size=n)
    Z2 = rand.uniform(int_par, 4, size=n)
    Z = Z1
    Z[indicator == 1] = Z2[indicator == 1]
    U1 = rand.normal(size=n)
    U2 = rand.normal(size=n)
    if oracle:
        gamma = 0
    else:
        gamma = 1
    X1 = 0.1 * Z + gamma * U1 * (Z <= 3.5) + 2 * Z * rand.normal(size=n)
    X2 = gamma * U2 + rand.normal(size=n)
    X = np.vstack([X1, X2]).T
    Y = f(X) + U1 + U2

    return X, Y, Z


def get_res_nonlinear(n, intvec, f, config_hsic, config_me):
    X, Y, Z = generate_data(n, 0.5, f)
    X_features = radial2D(X, centres)

    # # fit OLS
    ols_net = NNHSICX(input_dim=2,
                      lr=1e-2,
                      lmd=-99)
    ols_net = train_mse(ols_net, config_me, X, Y, Z)

    ols_basis = PredPolyRidge(1, alpha=0, bias=True)
    ols_basis.fit(X_features, Y)

    # fit AR
    AR_coefs = train_AR(X_features, Y, Z)

    # fit HSIC IV
    kernel_e = RBFKernel(sigma=1)
    kernel_z = RBFKernel(sigma=1)

    hsic_net = NNHSICX(input_dim=2,
                       lr=config_hsic['lr'],
                       lmd=config_hsic['lmd'],
                       kernel_e=kernel_e,
                       kernel_z=kernel_z)

    hsic_net.load_state_dict(ols_net)
    hsic_net = train_HSIC_IV(hsic_net, config_hsic, X, Y, Z)
    intercept_adjust = Y.mean() - hsic_net(X).mean()

    hsic_basis = LinearHSICX(input_dim=10,
                             lr=config_hsic['lr'],
                             lmd=config_hsic['lmd'],
                             kernel_e=kernel_e,
                             kernel_z=kernel_z)

    hsic_basis.load_state_dict(ols_basis.reg.coef_)
    hsic_basis = train_HSIC_IV(hsic_basis, config_hsic, X_features, Y, Z)
    intercept_adjust_basis = Y.mean() - hsic_basis(X_features).mean()

    df_res = None
    # evaluate on test distributions
    for i, iv in enumerate(intvec):
        X_test, Y_test, Z_test = generate_data(n, iv, f)
        X_features_test = radial2D(X_test, centres)
        # get predictions
        ols_pred = ols_net(X_test).detach().numpy()
        ols_pred_basis = ols_basis.predict(X_features_test)

        hsic_pred = intercept_adjust + hsic_net(X_test)
        hsic_pred = hsic_pred.detach().numpy()

        hsic_pred_basis = intercept_adjust_basis + hsic_basis(X_features_test)
        hsic_pred_basis = hsic_pred_basis.detach().numpy()

        causal_pred = f(X_test)
        AR_pred = AR_coefs[0] + AR_coefs[1:] @ X_features_test.T
        # compute loss
        ols_loss = np.mean((Y_test - ols_pred) ** 2)
        ols_loss_basis = np.mean((Y_test - ols_pred_basis) ** 2)
        hsic_loss = np.mean((Y_test - hsic_pred) ** 2)
        hsic_loss_basis = np.mean((Y_test - hsic_pred_basis) ** 2)
        causal_loss = np.mean((Y_test - causal_pred) ** 2)
        AR_loss = np.mean((Y_test - AR_pred) ** 2)

        inner_df = pd.DataFrame([[
            hsic_loss_basis,
            hsic_loss,
            ols_loss_basis,
            ols_loss,
            AR_loss,
            causal_loss,
            iv]],
            columns=[
                'HSIC-X-pen-BASIS',
                'HSIC-X-pen-NN',
                'OLS-BASIS',
                'OLS-NN',
                'AR-BASIS',
                'Causal',
                'IntStr'])
        if df_res is None:
            df_res = inner_df
        else:
            df_res = df_res.append(inner_df, ignore_index=True)

    return df_res


def radial2D(X, centres):
    Phi = np.zeros((X.shape[0], num_basis))
    for i in range(num_basis):
        Phi[:, i:i + 1] = np.exp(-1 * (np.linalg.norm(X - centres[i], axis=1, keepdims=True) / 3) ** 2)

    return Phi


if __name__ == "__main__":
    ## Experiment
    n = 3000
    B = 10
    intvec = [.5, 1, 1.5, 2, 2.5, 3, 3.5, 3.99]
    linear = False
    num_basis = 10

    np.random.seed(0)
    if linear:
        f = lambda x: x[:, 0] + x[:, 1]
    else:
        x_min = [-5, -5]
        x_max = [5, 5]
        centres = np.random.uniform(low=x_min, high=x_max, size=(num_basis, 2))
        w = np.random.normal(0, 4, size=num_basis)
        f = lambda x: radial2D(x, centres) @ w

    config_hsic = {'batch_size': 256,
                   'lr': 5e-3,
                   'max_epoch': 400,
                   'num_restart': 1,
                   'lmd': 1e-5}

    config_me = {'batch_size': 256,
                 'lr': 1e-2,
                 'max_epoch': 400}

    df_res = []
    for _ in range(B):
        df_res += [get_res_nonlinear(n, intvec, f, config_hsic, config_me)]

    df_res = functools.reduce(lambda df1, df2: df1.append(df2, ignore_index=True), df_res)
    melt_res_df = df_res.melt(id_vars=['IntStr'], var_name='Method', value_name='MSE')
    pd.to_csv(melt_res_df, "DG_compare_df_nonlin.csv", index=False)
    # plotting
    # loading linear result to plot together
    melt_lin = pd.read_csv("../results/DG_compare_df.csv")
    melt_lin[r'$f$'] = 'Linear'
    melt_res_df[r'$f$'] = 'Non-linear'

    melt_res_df['Function class'] = melt_res_df['Method']. \
        apply(lambda x: 'Known Basis' if (('BASIS' in x) or (x == 'Causal')) else 'NN')
    melt_lin['Function class'] = 'Known Basis'
    melt_res_df['Method'] = melt_res_df['Method'] \
        .str.replace("-BASIS", "") \
        .str.replace("-NN", "")
    melt_res_df = melt_lin.append(melt_res_df, ignore_index=True)

    sns.set(font_scale=1.7, style='white', palette=sns.set_palette("tab10"))

    g = sns.relplot(data=melt_res_df, kind='line', col=r'$f$',
                    x='IntStr', y='MSE', hue='Method', style=r'Function class',
                    hue_order=['AR', 'OLS', 'Causal', 'HSIC-X-pen'],
                    markers=True, aspect=1, height=3.6, ci=95, facet_kws={'sharey': False, 'sharex': True})

    g.set_xlabels(r"Intervention strength $(i)$")
    g.set_ylabels('MSE')

    for ax in g.axes.flat:
        ax.set_yscale('log', base=2)

    g.set(xticks=[0.5, 1.5, 2.5, 3.5], xticklabels=[0.5, 1.5, 2.5, 3.5])
    g.set_titles(r'$f^0$ = {col_name}')

    plt.xlabel(r"Intervention strength $(i)$")
    plt.savefig(
        'results/DG_compare_nonlin.pdf',
        bbox_inches="tight")
    plt.close()
