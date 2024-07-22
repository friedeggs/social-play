from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import matplotlib.ticker as tkr

import numpy as np

from typing import List

import pandas as pd
import seaborn as sns
sns.set_theme(style="white")
sns.set_context("paper", font_scale=3)
sns.set_palette("bright")

pp = PdfPages('figure_5-final.pdf')

# plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['font.family'] = 'monospace'


data = pd.DataFrame({'x': [
	# "Frozen", "ISR", "Oracle",
	# "Frozen", "ISR", "Oracle",
	# "Frozen", "ISR", "Oracle",
	# "Frozen", "ISR", "Oracle",
	# "Frozen", "ISR", "Oracle",
	# "Frozen", "ISR", "Oracle",


	"Frozen","Frozen","Frozen","Frozen","Frozen",
	"ISR","ISR","ISR","ISR","ISR",
	"Oracle","Oracle","Oracle","Oracle","Oracle",

	# "Frozen","Frozen","Frozen","Frozen","Frozen",
	# "ISR","ISR","ISR","ISR","ISR",
	# "Oracle","Oracle","Oracle","Oracle","Oracle",

	# "Frozen","Frozen","Frozen","Frozen","Frozen",
	# "ISR","ISR","ISR","ISR","ISR",
	# "Oracle","Oracle","Oracle","Oracle","Oracle",

	# "Frozen","Frozen","Frozen","Frozen","Frozen",
	# "ISR","ISR","ISR","ISR","ISR",
	# "Oracle","Oracle","Oracle","Oracle","Oracle",

	# "Frozen","Frozen","Frozen","Frozen","Frozen",
	# "ISR","ISR","ISR","ISR","ISR",
	# "Oracle","Oracle","Oracle","Oracle","Oracle",


	"Oracle","Oracle","Oracle","Oracle","Oracle",
	"ISR","ISR","ISR","ISR","ISR",
	"Frozen","Frozen","Frozen","Frozen","Frozen",


	"Oracle","Oracle","Oracle","Oracle","Oracle",
	"ISR","ISR","ISR","ISR","ISR",
	"Frozen","Frozen","Frozen","Frozen","Frozen",


	"Oracle","Oracle","Oracle","Oracle","Oracle",
	"ISR","ISR","ISR","ISR","ISR",
	"Frozen","Frozen","Frozen","Frozen","Frozen",


	"Oracle","Oracle","Oracle","Oracle","Oracle",
	"ISR","ISR","ISR","ISR","ISR",
	"Frozen","Frozen","Frozen","Frozen","Frozen",

#####


	"Oracle","Oracle","Oracle","Oracle","Oracle",
	"ISR","ISR","ISR","ISR","ISR",
	"Frozen","Frozen","Frozen","Frozen","Frozen",


	"Oracle","Oracle","Oracle","Oracle","Oracle",
	"ISR","ISR","ISR","ISR","ISR",
	"Frozen","Frozen","Frozen","Frozen","Frozen",


	"Oracle","Oracle","Oracle","Oracle","Oracle",
	"ISR","ISR","ISR","ISR","ISR",
	"Frozen","Frozen","Frozen","Frozen","Frozen",


	"Oracle","Oracle","Oracle","Oracle","Oracle",
	"ISR","ISR","ISR","ISR","ISR",
	"Frozen","Frozen","Frozen","Frozen","Frozen",


	"Oracle","Oracle","Oracle","Oracle","Oracle",
	"ISR","ISR","ISR","ISR","ISR",
	"Frozen","Frozen","Frozen","Frozen","Frozen",

],
'y': [
	# # -0.00135,0.3263,-0.197,-0.29535,
	# 0.41535,0.60415,0.6788,
	# 0.0391,0.42765,0.49,
	# 0.14025, 0.46965, 0.5295,
	# 0.24145,0.43815,0.59605,
	# 0.70645, 0.68605,0.86055,
	# # -0.00045,0.43015,0.62215,0.79065,
	# # -0.0045,0.92365,1.15725,1.18205,


	# 0.46625,0.5931,0.6824,
	# 0.17955,0.4064,0.5095,
	# 0.266, 0.45565, 0.53875,
	# 0.3692,0.46005,0.6161,
	# 0.6846, 0.69215,0.8809,

	# 0.3345,0.56575,0.53885,
	# 0.18515,0.39765,0.39495,
	# 0.3089,0.49485,0.47825,
	# 0.29075,0.4573,0.45345,
	# 0.793,0.6824,0.803,

0.3345,	0.41305,	0.6714,	0.45605,	0.443,
0.56575,	0.4382,	0.5676,	0.61315,	0.4105,
0.53885,	0.60235,	0.55725,	0.6205,	0.5664,

0.39495,	0.4245,	0.34085,	0.56645,	0.39055,
0.39765,	0.3681,	0.4032,	0.5374,	0.3884,
0.18515,	0.01815,	0.06405,	0.1942,	0.0353,

0.47825,	0.42895,	0.49305,	0.514,	0.403,
0.49485,	0.4295,	0.53735,	0.35395,	0.40255,
0.3089,	0.12005,	0.35025,	0.18715,	0.1517,

0.45345,	0.63215,	0.7251,	0.4992,	0.5479,
0.4573,	0.47335,	0.5482,	0.6496,	0.4509,
0.29075,	0.3925,	0.4093,	0.41405,	0.3423,

0.803	,0.86085,	0.90105,	0.8637,	0.78965,
0.6824,	0.7547,	0.68785,	0.86865,	0.83695,
0.793	,0.8191,	0.74515,	0.8468,	0.841,

# 2.08,	2.34,	2.79,	2.12,	1.94,
# 2.18,	2.47,	2.4,	2.56,	2.05,
# 2.56,	2.84,	2.4,	2.62,	2.64,
				
				
# 1.23,	1.63,	1.5,	1.57,	1.31,
# 1.89,	1.95,	1.98,	2.5,	1.97,
# 2.04,	2.35,	1.85,	2.54,	1.99,
				
				
# 1.55,	1.74,	2.04,	1.72,	1.61,
# 2.03,	2.1,	2.33,	2.2,	2.02,
# 2.05,	2.46,	2.1,	2.41,	2.02,
				
				
# 1.96,	1.93,	2.2,	1.88,	1.56,
# 2.08,	2.01,	2.32,	2.65,	2.16,
# 2.63,	2.56,	2.94,	2.22,	2.56,
				
				
# 3.13,	2.8,	2.98,	3.17,	3.16,
# 3.05,	2.6,	2.56,	3.21,	3.16,
# 3.24,	3.28,	3.3,	3.21,	3.08,

0.51755,0.5435,0.5633,0.6466500000000001,0.7659999999999999,
0.4919,0.4356499999999999,0.5809000000000001,0.5974,0.638,
0.25730000000000003,0.34745,0.17964999999999992,0.36845000000000006,0.5684,

0.34965,0.4989500000000001,0.44755,0.3670999999999999,0.5428999999999999,
0.358,0.48550000000000004,0.41835000000000006,0.4949000000000001,0.47769999999999996,
-0.25860000000000005,0.16325,0.053099999999999994,0.11305000000000004,0.19845,

0.40175000000000005,0.41970000000000013,0.46130000000000004,0.60565,0.6378499999999999,
0.3375,0.3814500000000001,0.42595,0.6666,0.5045499999999999,
0.07119999999999999,0.22950000000000004,0.13935,0.26385,0.37775000000000014,

0.37715,0.5594,0.49255000000000004,0.6504000000000001,0.7802999999999999,
0.48300000000000004,0.48335000000000006,0.5245000000000001,0.6496999999999999,0.5706,
0.28095000000000003,0.36985000000000007,0.16845000000000002,0.40445000000000014,0.5948499999999999,

0.69635,0.8396000000000001,0.8435500000000001,0.929,0.888,
0.6732,0.8046500000000002,0.6285500000000002,0.88605,0.71155,
0.76625,0.82365,0.81955,0.7911499999999999,0.9011,



],
'env': [
	# "Halves","Halves","Halves","Halves",

	"\"plus\"","\"plus\"","\"plus\"","\"plus\"","\"plus\"",
	"\"plus\"","\"plus\"","\"plus\"","\"plus\"","\"plus\"",
	"\"plus\"","\"plus\"","\"plus\"","\"plus\"","\"plus\"",
	"\"x\"","\"x\"","\"x\"","\"x\"","\"x\"",
	"\"x\"","\"x\"","\"x\"","\"x\"","\"x\"",
	"\"x\"","\"x\"","\"x\"","\"x\"","\"x\"",
	"\"U\"","\"U\"","\"U\"","\"U\"","\"U\"",
	"\"U\"","\"U\"","\"U\"","\"U\"","\"U\"",
	"\"U\"","\"U\"","\"U\"","\"U\"","\"U\"",
	"\"T\"","\"T\"","\"T\"","\"T\"","\"T\"",
	"\"T\"","\"T\"","\"T\"","\"T\"","\"T\"",
	"\"T\"","\"T\"","\"T\"","\"T\"","\"T\"",
	"4 rooms","4 rooms","4 rooms","4 rooms","4 rooms",
	"4 rooms","4 rooms","4 rooms","4 rooms","4 rooms",
	"4 rooms","4 rooms","4 rooms","4 rooms","4 rooms",

	"\"plus\"","\"plus\"","\"plus\"","\"plus\"","\"plus\"",
	"\"plus\"","\"plus\"","\"plus\"","\"plus\"","\"plus\"",
	"\"plus\"","\"plus\"","\"plus\"","\"plus\"","\"plus\"",
	"\"x\"","\"x\"","\"x\"","\"x\"","\"x\"",
	"\"x\"","\"x\"","\"x\"","\"x\"","\"x\"",
	"\"x\"","\"x\"","\"x\"","\"x\"","\"x\"",
	"\"U\"","\"U\"","\"U\"","\"U\"","\"U\"",
	"\"U\"","\"U\"","\"U\"","\"U\"","\"U\"",
	"\"U\"","\"U\"","\"U\"","\"U\"","\"U\"",
	"\"T\"","\"T\"","\"T\"","\"T\"","\"T\"",
	"\"T\"","\"T\"","\"T\"","\"T\"","\"T\"",
	"\"T\"","\"T\"","\"T\"","\"T\"","\"T\"",
	"4 rooms","4 rooms","4 rooms","4 rooms","4 rooms",
	"4 rooms","4 rooms","4 rooms","4 rooms","4 rooms",
	"4 rooms","4 rooms","4 rooms","4 rooms","4 rooms",

	# "Walls","Walls","Walls","Walls",
	# "Lava","Lava","Lava","Lava",
]
})

# import pdb; pdb.set_trace()
print()
ax = sns.catplot(x="x", y="y", col="env", kind="bar", sharey=True, errorbar=('ci', 68), # yerr=[[0],[0]],
                  data=data, hue='x', hue_order=["Oracle", "Frozen", "None", "ISR"])
# ax = sns.lineplot(x="x", y="y", 
# 				  data=data, alpha=0.5, color='blue', label='')
# ax = sns.lineplot(x="x", y="y2", 
#                   data=data, alpha=0.5, color='blue', label='')
# ax.axvline(x = 4, ymin = 0, ymax = 1, color='black')
# plt.legend()
# ax.set(xlabel='Method', ylabel='Reward')
ax.set(xlabel='', ylabel='Reward (OOD)')
# ax.set_titles("{col_name}")
ax.set_titles("")
# ax.legend.set_title(None)
ax.legend.set_visible(False)
# print(dir(ax.legend))
# ax.fig.suptitle('Handpicked Generalizations', y=1.05)
# plt.title("Generalization")


# x = ep_rew_mean['Step']
# ymin = ep_rew_mean['minigrid_figure_3a - log_reward_model/Loss/train__MIN']
# ymax = ep_rew_mean['minigrid_figure_3a - log_reward_model/Loss/train__MAX']
# ax.fill_between(x, ymin, ymax, color='blue', alpha=0.25)
# ax.xaxis.set_ticklabels([0, 2, 4, 8, 10, 12])
# ax.xaxis.set_ticks([0, 30, 60, 90, 120])
# ax.xaxis.set_ticklabels([0, 0.1, 0.5, 2])
# ax.xaxis.set_ticks([0, 0.1, 0.5, 2])
# ax.set(xlabel='Environment', ylabel='Reward')
# plt.yscale("log")
# ax.yaxis.set_ticks([5e-4, 5.4e-4, 5.8e-4])
# ax.yaxis.set_ticklabels([5e-4, 5.4e-4, 5.8e-4])
# ax.yaxis.set_ticks([0, .5, 1.])
for _ax in ax.axes.flat:
    _ax.yaxis.set_ticks([0, .5, 1.])
    # _ax.yaxis.set_ticks([0, 1.8, 3.6])
#     # _ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda y, p: f'{y:.2f}'))
#     _ax.tick_params(axis='y', which='minor', bottom=False)
#     # _ax.margins(0.05) 
# ax.tick_params(axis='y', which='minor', bottom=False)

# ax.set_ylim(0, 1)
# plt.axis('scaled')
# plt.axis('square')
# plt.gca().set_aspect(2)
# plt.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0,
#                             wspace=0)
# plt.tight_layout()
# plt.tight_layout(pad=0, w_pad=1, h_pad=1)
# plt.gca().autoscale_view('tight')
# plt.axis('equal')
# plt.gca().set_aspect('equal', adjustable='box')
# ax.set_aspect('equal', adjustable='box')
# ax_nr.set_aspect('equal', adjustable='box')
# sns.despine()#right = True, left = True)
pp.savefig(bbox_inches='tight')

pp.close()

# import statsmodels.api as sm
# from statsmodels.formula.api import ols

# # dta = sm.datasets.longley.load_pandas().data
# # # import pdb; pdb.set_trace()
# # formula = 'TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR'
# # results = ols(formula, dta).fit()
# # hypotheses = 'GNPDEFL = GNP, UNEMP = 2, YEAR = 1829'
# # t_test = results.t_test(hypotheses)
# # print(t_test)

# formula = 'y ~ x * env'
# results = ols(formula, data).fit()
# print(dir(results))
# print(results.summary())
# # hypotheses = 'x[T.Reward] = x[T.Simple]'
# # t_test = results.t_test(hypotheses)
# # print(t_test)

