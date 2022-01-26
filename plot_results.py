import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

for fn in ['linear', 'radial']:
    for instrument in ['Gaussian', 'Binary']:
        final_df = pd.read_csv("Ind-IV/results/compare_df_fn_{}_ins_{}.csv".format(fn, instrument))

        sns.set(font_scale=1.4, style='white', palette=sns.set_palette("tab10"))

        g = sns.catplot(data=final_df, kind="point", log=True,
                        x='alpha', y='MISE', hue='Method',
                        markers=["o", "x", "d", "s"], linestyles=[':', '--', '-', '-.'],
                        capsize=.07, aspect=1.5, height=3.2, ci=95)
        g.fig.get_axes()[0].set_yscale('log')
        g._legend.remove()

        plt.xlabel(r'$\alpha$')
        plt.ylabel("MSE")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=4, fancybox=True, shadow=False, prop={'size': 11})

        plt.savefig('results/compare_alpha_fn_{}_ins_{}.pdf'.format(fn, instrument),
                    bbox_inches="tight")
        plt.close()
