from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

import numpy as np

from typing import List

import pandas as pd
import seaborn as sns
import scipy
sns.set_theme(style="white")
sns.set_context("paper", font_scale=2.5)
sns.set_palette("bright")

pp = PdfPages('figure_4b-final.pdf')

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['figure.figsize'] = (4, 5)

data = pd.DataFrame({'x': [
	# "None", 
	# "Frozen", "ISR", "Oracle",
	# "None", 
	# "Frozen", "ISR", "Oracle",
	# "None", 
	# "Frozen", "ISR", "Oracle",

	"Frozen", "Frozen", "Frozen","Frozen", "Frozen", "Frozen", "Frozen", "Frozen", "Frozen","Frozen", 
	# "Frozen", "Frozen", "Frozen","Frozen", "Frozen", "Frozen","Frozen", "Frozen", "Frozen","Frozen",
	"ISR", "ISR", "ISR","ISR", "ISR", "ISR","ISR", "ISR", "ISR","ISR",
	"Oracle", "Oracle", "Oracle","Oracle", "Oracle", "Oracle","Oracle", "Oracle", "Oracle","Oracle", 

],
'y': [
	# -0.0045,0.92365,1.15725,1.18205,
	# -0.00045,0.43015,0.62215,0.79065,
	# -0.0105,0.14845,0.2879,0.3584,

	# 1.116,1.13545,1.229,
	# 0.51655,0.67805,0.57865,

	# 0.5508166667,0.6108666667,0.66765,

# 0.6545,	0.57865,	0.7698,
# 0.54755,	0.67805,	0.607,
# 0.46815,	0.51655,	0.66775,

0.46815,	0.51655,	0.66775,	0.5002,	0.61165,	0.4882,	0.41905,	0.65745,	0.56195,	0.515, 
# 0.47545,	0.48705,	0.3728,	0.4349,	0.4927,	0.37725,	0.5317,	0.5898,	0.6083,	0.4169,
0.54755,	0.67805,	0.607,	0.81355,	0.61925,	0.5173,	0.5994,	0.59855,	0.71735,	0.6025,
0.6545,	0.57865,	0.7698,	0.8218,	0.75765,	0.513,	0.60075,	0.615,	0.70365,	0.7042,

# 2.09,	2.29,	2.48,	2.38,	2.1,	1.99,	2.26,	2.28,	2.46,	2.24,	
# 2.6,	2.46,	2.44,	2.34,	2.31,	2.14,	2.23,	2.5,	2.21,	2.79,
# 2.32,	2.57,	2.61,	2.58,	2.9,	2.23,	2.32,	3.02,	2.8,	2.3,

],
# 'env': [
# 	# "Halves",
# 	# "Halves","Halves","Halves",
# 	# "Walls",
# 	"Walls","Walls","Walls",
# 	# "Lava",
# 	# "Lava","Lava","Lava",
# ]
})

# import pdb; pdb.set_trace()

# ax = sns.catplot(x="x", y="y", col="env", kind="bar", sharey=True,
ax = sns.catplot(x="x", y="y", kind="bar", errorbar=('ci', 68), 
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
# ax.fig.suptitle('Generalization', y=1.05)
# plt.title("Generalization")

def make_square_axes(ax):
    """Make an axes square in screen units.

    Should be called after plotting.
    """
    ax.set_aspect(.85 / ax.get_data_ratio())

make_square_axes(plt.gca())


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
for _ax in ax.axes.flat:
    _ax.yaxis.set_ticks([0, 0.4, 0.8])
    # _ax.yaxis.set_ticks([0, .5, 1, 1.5, 2, 2.5])
    # _ax.yaxis.set_ticks([.5, 1.5, 2.5])
# ax.yaxis.set_ticklabels([0, .2, .4, .6, .8])
# ax.yaxis.set_ticklabels([5e-4, 5.4e-4, 5.8e-4])
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
sns.despine()#right = True, left = True)
pp.savefig(bbox_inches='tight')
pp.close()

# rvs1 = data['y'][0:20]
# rvs2 = data['y'][20:30]
# rvs3 = data['y'][30:40]
rvs1 = data['y'][0:10]
rvs2 = data['y'][10:10+10]
rvs3 = data['y'][10+10:]

print(scipy.stats.ttest_ind(rvs1, rvs2))
print(scipy.stats.ttest_ind(rvs2, rvs3))
