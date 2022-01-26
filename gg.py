# for instrument in ['Gaussian', 'Binary']:
for instrument in ['Gaussian', 'Binary']:
    res_df = None
    # get a fix x_vis
    iv_type = 'mix_{}'.format(instrument)
    _, _, _, X_vis = gen_data(f, n, iv_type)

    alphas = np.linspace(0, 1, 5)

    for j in range(len(alphas)):
        alpha = alphas[j]

        def rep_function(i):
            X, Y, Z, _ = gen_data(f, n, iv_type, alpha=alpha)
            # dev set for DeepGMM
            X_dev, Y_dev, Z_dev, _ = gen_data(f, n, iv_type, alpha=alpha)
            # oracle set
            X_o, Y_o, Z_o, _ = gen_data(f, n, iv_type, alpha=alpha, oracle=True)
            X_test, _, _, _ = gen_data(f, X_vis.shape[0], iv_type, alpha=alpha)

            # Pure predictive
            mse_net = NonlinearModel(input_dim=1,
                                      lr=config_mse['lr'],
                                      lmd=-99)

            mse_net = train_mse(mse_net, config_mse, X, Y, Z)
            y_hat_mse = mse_net(to_torch(X_test)).detach().numpy()

            oracle_net = train_mse(mse_net, config_mse, X_o, Y_o, Z_o)
            y_hat_oracle = mse_net(to_torch(X_test)).detach().numpy()

            # HSIC IV
            s_z = med_sigma(Z)
            kernel_e = RBFKernel(sigma=1)

            if instrument == 'Binary':
                kernel_z = CategoryKernel()
            else:
                kernel_z = RBFKernel(sigma=s_z)

            # non regularized HSIC IV
            hsic_net = NonlinearModel(input_dim=1,
                                      lr=config_hsic['lr'],
                                      kernel_e=kernel_e,
                                      kernel_z=kernel_z)

            hsic_net.load_state_dict(mse_net)
            hsic_net = train_HSIC_IV(hsic_net, config_hsic, X, Y, Z, verbose=False)

            intercept_adjust = Y.mean() - hsic_net(to_torch(X)).mean()
            y_hat_hsic = intercept_adjust + hsic_net(to_torch(X_test))
            y_hat_hsic = y_hat_hsic.detach().numpy().copy()

            # regularized HSIC IV
            hsic_net = NonlinearModel(input_dim=1,
                                      lr=config_hsic['lr'],
                                      kernel_e=kernel_e,
                                      kernel_z=kernel_z,
                                      lmd=5e-5)

            hsic_net.load_state_dict(mse_net)
            hsic_net = train_HSIC_IV(hsic_net, config_hsic, X, Y, Z, verbose=False)

            intercept_adjust = Y.mean() - hsic_net(to_torch(X)).mean()
            y_hat_hsic_pen = intercept_adjust + hsic_net(to_torch(X_test))
            y_hat_hsic_pen = y_hat_hsic_pen.detach().numpy().copy()

            # prepare data for DeepGMM
            dat = [X, Z, Y, X_dev, Z_dev, Y_dev]
            # to torch
            for k in range(len(dat)):
                dat[k] = to_torch(dat[k]).double()

            deepGMM = ToyModelSelectionMethod()
            deepGMM.fit(*dat, g_dev=None, verbose=True)
            y_hat_deepGMM = deepGMM.predict(to_torch(X_test).double()).flatten().detach().numpy()

            inner_df = pd.DataFrame()
            inner_df_vis = pd.DataFrame()

            inner_df['f_x'] = f(X_test)
            inner_df['Pred'] = y_hat_mse
            inner_df['HSIC-IV'] = y_hat_hsic
            inner_df['HSIC-AR'] = y_hat_hsic_pen
            inner_df['D-GMM'] = y_hat_deepGMM
            inner_df['Oracle'] = y_hat_oracle
            inner_df['alpha'] = alpha
            inner_df['run_id'] = i

            return inner_df

        ret_df = Parallel(n_jobs=4)(
            delayed(rep_function)(i=i) for i in range(n_rep))

        # ret_df = [rep_function(i) for i in range(n_rep)]

        ret_df = functools.reduce(lambda df1, df2: df1.append(df2, ignore_index=True), ret_df)

        if res_df is None:
            res_df = ret_df
        else:
            res_df = res_df.append(ret_df, ignore_index=True)

    melt_res_df = res_df.melt(id_vars=['f_x', 'alpha', 'run_id'], var_name='Method',
                              value_name='y_pred')
    melt_res_df['MISE'] = (melt_res_df['f_x'] - melt_res_df['y_pred']) ** 2
    final_df = melt_res_df.groupby(['Method', 'alpha', 'run_id'])['MISE'].mean().reset_index()
    final_df['alpha'] = np.round(final_df.alpha, 2)
    final_df.to_csv("results/compare_df_NN_ins_{}.csv".format(instrument),
                    index=False)
    sns.set(font_scale=1.4, style='white', palette=sns.set_palette("tab10"))

    g = sns.catplot(data=final_df, kind="point", log=True,
                    x='alpha', y='MISE', hue='Method',
                    markers=["o", "x", "d", "s", "v"], linestyles=[':', '--', '-', '-.', ':'],
                    capsize=.07, aspect=1.5, height=3.2, ci=95)
    g.fig.get_axes()[0].set_yscale('log')
    g._legend.remove()

    plt.xlabel(r'$\alpha$')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=5, fancybox=True, shadow=False, prop={'size': 10})
    plt.savefig('results/compare_NN_ins_{}.pdf'.format(instrument),
                bbox_inches="tight")
    plt.close()
