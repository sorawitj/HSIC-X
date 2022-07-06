import functools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from helpers.trainer import train_HSIC_IV
from models.hsicx import LinearHSICX
from models.kernel import RBFKernel
from models.baselines import PredPolyRidge
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
import rpy2.robjects.packages as packages

pandas2ri.activate()
numpy2ri.activate()

AR = packages.importr("AnchorRegression")


# define anchor regression function
def train_AR(X, Y, Z):
    df_X = pd.DataFrame(X, columns=['X' + str(i) for i in range(X.shape[1])])
    df_X['Y'] = Y
    df_Z = pd.DataFrame(Z, columns=['Z'])
    coefs = AR.anchor_regression(df_X, df_Z, 1e2, 'Y', 0.).rx2('coeff')
    return coefs


def generate_data(n, int_par, causal_par):
    rand = np.random
    indicator = rand.binomial(1, int_par / 4, size=n)
    Z1 = rand.uniform(0, int_par, size=n)
    Z2 = rand.uniform(int_par, 4, size=n)
    Z = Z1
    Z[indicator == 1] = Z2[indicator == 1]
    U1 = rand.normal(size=n)
    U2 = rand.normal(size=n)
    X1 = 0.1 * Z + U1 * (Z <= 3.5) + 2 * Z * rand.normal(size=n)
    X2 = U2 + rand.normal(size=n)
    X = np.vstack([X1, X2])
    Y = causal_par @ X + U1 + U2

    return X.T, Y, Z


def get_res(n, intvec, causal_par, config):
    X, Y, Z = generate_data(n, 0.5, causal_par)

    # fit OLS
    ols = PredPolyRidge(degree=1, alpha=0., bias=True)
    ols.fit(X, Y)
    # fit HSIC IV
    kernel_e = RBFKernel(sigma=1)
    kernel_z = RBFKernel(sigma=1)
    # fit Anchor Regression
    AR_coefs = train_AR(X, Y, Z)

    hsic_net = LinearHSICX(input_dim=2,
                           lr=config['lr'],
                           lmd=config['lmd'],
                           kernel_e=kernel_e,
                           kernel_z=kernel_z,
                           bias=False)

    hsic_net.load_state_dict(ols.reg.coef_.flatten())
    hsic_net = train_HSIC_IV(hsic_net, config, X, Y, Z)
    intercept_adjust = Y.mean() - hsic_net(X).mean()

    df_res = None
    # evaluate on test distributions
    for i, iv in enumerate(intvec):
        X_test, Y_test, _ = generate_data(n, iv, causal_par)
        # get predictions
        ols_pred = ols.predict(X_test)

        hsic_pred = intercept_adjust + hsic_net(X_test)
        hsic_pred = hsic_pred.detach().numpy()

        causal_pred = causal_par @ X_test.T
        AR_pred = AR_coefs[0] + AR_coefs[1:] @ X_test.T
        # compute loss
        ols_loss = np.mean((Y_test - ols_pred) ** 2)
        hsic_loss = np.mean((Y_test - hsic_pred) ** 2)
        causal_loss = np.mean((Y_test - causal_pred) ** 2)
        AR_loss = np.mean((Y_test - AR_pred) ** 2)

        inner_df = pd.DataFrame([[AR_loss, hsic_loss, ols_loss, causal_loss, iv]],
                                columns=['AR', 'HSIC-X-pen', 'OLS', 'Causal', 'IntStr'])
        if df_res is None:
            df_res = inner_df
        else:
            df_res = df_res.append(inner_df, ignore_index=True)

    return df_res


if __name__ == "__main__":
    ## Experiment
    n = 3000
    B = 10
    np.random.seed(0)

    intvec = [.5, 1, 1.5, 2, 2.5, 3, 3.5, 3.99]
    causal_par = np.array([1, 1])

    config = {'batch_size': 256,
              'lr': 5e-3,
              'max_epoch': 400,
              'num_restart': 1,
              'lmd': 1e-4}

    df_res = []
    for _ in range(B):
        df_res += [get_res(n, intvec, causal_par, config)]

    df_res = functools.reduce(lambda df1, df2: df1.append(df2, ignore_index=True), df_res)

    # plotting
    melt_res_df = df_res.melt(id_vars=['IntStr'], var_name='Method', value_name='MSE')
    melt_res_df.to_csv("../results/DG_compare_df.csv", index=False)

    sns.set(font_scale=1.4, style='white', palette=sns.set_palette("tab10"))

    g = sns.relplot(data=melt_res_df, kind='line',
                    x='IntStr', y='MSE', hue='Method', style='Method',
                    hue_order=['AR', 'OLS', 'Causal', 'HSIC-X-pen'],
                    markers=True, aspect=1.5, height=2.7, ci=95)

    plt.legend(loc='upper center', bbox_to_anchor=(0.4, 1.12),
               ncol=4, fancybox=True, shadow=True, prop={'size': 12})
    plt.yscale('log')
    plt.xlabel(r"Intervention strength $(i)$")
    plt.savefig(
        '../results/DG_compare.pdf',
        bbox_inches="tight")
    plt.close()
