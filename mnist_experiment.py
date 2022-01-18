from torch.utils.data import TensorDataset

from DeepGMM.scenarios.abstract_scenario import AbstractScenario
from hsic import *
from kernel import PolyKernel, RBFKernel, CategoryKernel
from model import LinearModel, MedianHeuristic, Poly2SLS, PvalueLog, PolyModel, NonlinearModel, CNNModel
import pytorch_lightning as pl
import pandas as pd
from utils import med_sigma, to_torch, gen_data, fit_restart
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

n_rep = 1

res_df = pd.DataFrame(columns=['x_vis', 'f_x', 'y_mse', 'y_hsic', 'alpha', 'run_id'])
# get a fix x_vis
iv_type = 'diffcoef'
np.random.seed(0)

scenario_name = "mnist_x"
scenario = AbstractScenario(filename="DeepGMM/data/" + scenario_name + "/main.npz")
scenario.to_2d()
scenario.info()

train = scenario.get_dataset("train")
dev = scenario.get_dataset("dev")
test = scenario.get_dataset("test")

x_dim = train.x.shape[1]

for i in range(n_rep):
    batch_size = 256  # 256
    trainloader = torch.utils.data.DataLoader(TensorDataset(to_torch(train.x),
                                                            to_torch(train.y),
                                                            to_torch(train.z)),
                                              batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    mse_net = CNNModel(input_dim=x_dim, lmd=0, lr=1e-3)
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(mse_net, trainloader)
    # get y_hat for MSE loss
    y_hat_mse = mse_net(to_torch(test.x)).detach().numpy()
    mse_direct = float(((y_hat_mse - test.g) ** 2).mean())

    s_z = med_sigma(train.z)
    # kernels = PolyKernel(degree=1)
    kernel_e = RBFKernel(sigma=1)
    # kernel_z = CategoryKernel()
    kernel_z = RBFKernel(sigma=s_z)
    # hsic_net = NonlinearModel(1, 5e3, 1e-2, kernel_e, kernel_z)
    hsic_net = CNNModel(input_dim=x_dim, lmd=-99, lr=1e-3,
                        kernel_e=kernel_e,
                        kernel_z=kernel_z)

    hsic_net.load_state_dict(mse_net)
    se_callback = MedianHeuristic()
    pval_callback = PvalueLog(B=1)

    hsic_net = fit_restart(trainloader, hsic_net, pval_callback, 10, se_callback, num_restart=1)
    y_hat_hsic = hsic_net(to_torch(test.x)).detach().numpy()
    mse_hsic = float(((y_hat_hsic - test.g) ** 2).mean())

    inner_df = pd.DataFrame(columns=res_df.columns)
    inner_df['f_x'] = test.g
    inner_df['y_mse'] = y_hat_mse
    inner_df['y_hsic'] = y_hat_hsic
    inner_df['run_id'] = i

    res_df = res_df.append(inner_df, ignore_index=True)

if True:
    melt_res_df = res_df.melt(id_vars=['x_vis', 'f_x', 'alpha', 'run_id'], var_name='model', value_name='y_pred')
    melt_res_df['Mean Square Error'] = (melt_res_df['f_x'] - melt_res_df['y_pred']) ** 2
    final_df = melt_res_df.groupby(['model', 'alpha', 'run_id'])['Mean Square Error'].mean().reset_index()
    final_df.to_csv("alpha_df3.csv", index=False)
    final_df['alpha'] = np.round(final_df.alpha, 2)
    g = sns.boxplot(data=final_df, x='alpha', y='Mean Square Error', hue='model')
    g.set_yscale("log")
    plt.title("Model: Quadratic, Instrument: Gaussian, without zero-mean regularization")
    plt.ylim(1e-2, 1e4)
    plt.savefig('results/compare_alpha_mean_gaussian.pdf')

if False:
    melt_res_df = res_df.melt(id_vars=['x_vis', 'alpha', 'run_id'], var_name='model', value_name='y')
    X, Y, _, _ = gen_data(f, n, iv_type)
    sns.scatterplot(X, Y, color='.5', linewidth=0, alpha=0.5)
    sns.lineplot(data=melt_res_df.query("model!='f_x'"), x="x_vis", y="y", units='run_id',
                 hue='model', estimator=None, lw=1, alpha=0.5,
                 palette=["blue", "red", "green"])
    df_0 = res_df.query("run_id == 0.0")
    plt.plot(df_0['x_vis'].values, df_0['f_x'].values, '--',
             label='f_x', c='black', lw=1.8)
    plt.legend()
    plt.ylim(Y.min() * 1.2, Y.max() * 1.2)
    plt.title("Model: Quadratic, Instrument: Variance")
    # plt.savefig('results/QIV_Var_BinaryZ.pdf')
