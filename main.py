from torch.utils.data import TensorDataset
from hsic import *
from model import LinearModel, MedianHeuristic, PolyModel, Poly2SLS, RBFKernel
import pytorch_lightning as pl
import pandas as pd
from utils import med_sigma, to_torch,  gen_data

n_rep = 20
n = 2000
# f = lambda x: 2*(1 / (1 + np.exp(-2 * x)))
# f = lambda x: np.log(np.abs(16. * x - 0) + 1) * np.sign(x - 0)
# f = lambda x: 3 * x
f = lambda x: 2 * x + .5 * x ** 2
res_df = pd.DataFrame(columns=['x_vis', 'f_x', 'y_mse', 'y_hsic'])

# get a fix x_vis
iv_type = 'diffcoef'
_, _, _, X_test = gen_data(f, n, iv_type)

for i in range(n_rep):
    X, Y, Z, _ = gen_data(f, n, 'diffcoef')

    batch_size = 256  # 256
    trainloader = torch.utils.data.DataLoader(TensorDataset(to_torch(X), to_torch(Y), to_torch(Z)),
                                              batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    # mse_net = PolyModel(input_dim=1, degree=2, lr=1e-2)
    # mse_net = NonlinearModel(1, lr=1e-3)
    mse_net = LinearModel(1, lr=1e-1)
    trainer = pl.Trainer(max_epochs=200)
    trainer.fit(mse_net, trainloader)
    # get y_hat for MSE loss
    y_hat_mse = mse_net(to_torch(X_test))

    s_z = med_sigma(Z)
    # kernels = PolyKernel(s_z=s_z)
    kernels = RBFKernel(s_z=s_z)
    hsic_net = PolyModel(input_dim=1,
                         degree=2,
                         lr=5e-2,  # lr=1e-3 doesn't seem to work!
                         lmd=-99,
                         kernels=kernels,
                         bias=False)

    # hsic_net.load_state_dict(mse_net.state_dict())
    se_callback = MedianHeuristic()
    # pval_callback = PvalueLog(B=50)
    trainer = pl.Trainer(max_epochs=200, callbacks=[se_callback])
    trainer.fit(hsic_net, trainloader)
    y_hat_hsic = hsic_net(to_torch(X_test))

    # 2SLS
    poly2SLS = Poly2SLS(degree=2)
    poly2SLS.fit(X, Y, Z)

    inner_df = pd.DataFrame(columns=['x_vis', 'f_x', 'y_mse', 'y_hsic'])
    inner_df['x_vis'] = X_test
    inner_df['f_x'] = f(X_test)
    inner_df['y_mse'] = y_hat_mse.detach().numpy()
    inner_df['y_hsic'] = y_hat_hsic.detach().numpy()
    inner_df['y_2sls'] = poly2SLS.predict(X_test)
    inner_df['run_id'] = i

    res_df = res_df.append(inner_df, ignore_index=True)

import seaborn as sns
import matplotlib.pyplot as plt

melt_res_df = res_df.melt(id_vars=['x_vis', 'run_id'], var_name='model', value_name='y')
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
plt.savefig('results/QIV_Var_BinaryZ.pdf')
