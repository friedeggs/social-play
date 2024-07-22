from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

from typing import List

import pandas as pd
import seaborn as sns
sns.set_theme(style="white")
sns.set_context("paper", font_scale=2)
sns.set_palette("bright")

pp = PdfPages('figure_6.pdf')

# plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['font.family'] = 'monospace'
# plt.rcParams["figure.figsize"] = (9,3)

# ep_rew_mean = pd.read_csv('wandb_export_2023-12-27T04 05 40.439-05 00.csv')
# ep_rew_mean_extr = pd.read_csv('wandb_export_2023-12-27T04 05 23.104-05 00.csv')
ep_rew_mean = pd.read_csv('wandb_export_2024-01-02T02 27 23.887-05 00.csv')
ep_rew_mean_extr = pd.read_csv('wandb_export_2024-01-02T02 27 03.970-05 00.csv')

print(ep_rew_mean['Step'])
print(ep_rew_mean.keys())
print(ep_rew_mean_extr.keys())

def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

keys = ep_rew_mean.keys()
ep_rew_mean[keys[0]] = smooth(ep_rew_mean[keys[0]], 0.8)
ep_rew_mean[keys[1]] = smooth(ep_rew_mean[keys[1]], 0.8)
ep_rew_mean[keys[2]] = smooth(ep_rew_mean[keys[2]], 0.8)

keys = ep_rew_mean_extr.keys()
ep_rew_mean_extr[keys[0]] = smooth(ep_rew_mean_extr[keys[0]], 0.8)
ep_rew_mean_extr[keys[1]] = smooth(ep_rew_mean_extr[keys[1]], 0.8)
ep_rew_mean_extr[keys[2]] = smooth(ep_rew_mean_extr[keys[2]], 0.8)


# import pdb; pdb.set_trace()
fig, ax = plt.subplots(1,2, figsize=(6,3))
# sns.lineplot(x="Step", y="minigrid_figure_5-walls-handpicked-reward-model_env-2-1235-1 - rollout/ep_rew_mean", 
# 			 data=ep_rew_mean[:60], label='With social reward', ax=ax[0])
# sns.lineplot(x="Step", y="minigrid_figure_5-walls-handpicked-reward-model_env-2-1235-1 - rollout/ep_rew_mean", 
# 			 data=ep_rew_mean[80:], label='', ax=ax[0], color='black')
# sns.lineplot(x="Step", y="minigrid_figure_5-walls-handpicked-reward-model_env-2-1235-1 - rollout/ep_rew_mean_extrinsic", 
# 			 data=ep_rew_mean_extr[:60], label='', ax=ax[1])
# sns.lineplot(x="Step", y="minigrid_figure_5-walls-handpicked-reward-model_env-2-1235-1 - rollout/ep_rew_mean_extrinsic", 
# 			 data=ep_rew_mean_extr[80:], label='', ax=ax[1], color='black')
# ax = sns.lineplot(x="x", y="y2", 
#                   data=data, alpha=0.5, color='blue', label='')
# ax.axvline(x = 4, ymin = 0, ymax = 1, color='black')
# ax[0].legend()
# ax[0].legend(prop={'size': 12})
# ax[0].legend(loc=[.15,.05], prop={'size': 16})
# ax[0].set(xlabel='Step', ylabel='Intrinsic Reward')
# ax[1].set(xlabel='Step', ylabel='Extrinsic Reward')
# ax[0].set_xlabel('Episodes (x $10^3$)', size=16)
# ax[0].set_ylabel('Internal reward', size=16)
# ax[1].set_xlabel('Episodes (x $10^3$)', size=16)
# # print(dir(ax[0].xaxis.label._x))
# # print(ax[0].xaxis.label._y)
# # print(ax[1].xaxis.label._y)
# print(ax[0].yaxis.label._x)
# print(ax[0].yaxis.label._y)
# # print(ax[1].xaxis.get_label_position())
# ax[0].yaxis.set_label_coords(-.35, 3.)
# ax[1].xaxis.set_label_coords(.35, -0.19)
# ax[1].set_ylabel('Goals reached', size=16)
# ax[1].yaxis.set_ticks([0.3, 0.35, 0.4])
# ax[1].yaxis.set_ticklabels([0.75, 0.875, 1])
# # ax[0].set_xlim([80., 180.])
# print(ax[1].get_xlim())
# ax[0].set_xlim(ax[1].get_xlim())
# ax[0].xaxis.set_ticks([100, 150])
# ax[1].xaxis.set_ticks([100, 150])
# ax[0].xaxis.set_ticklabels([10, 15])
# ax[1].xaxis.set_ticklabels([10, 15])

# divider = make_axes_locatable(ax[0])
# ax2 = divider.new_vertical(size="600%", pad=0.1)
# fig.add_axes(ax2)

# sns.lineplot(x="Step", y="minigrid_figure_5-walls-handpicked-reward-model_env-2-1235-1 - rollout/ep_rew_mean", 
# 			 data=ep_rew_mean[80:], label='', ax=ax2, color='black')
# ax2.set_ylabel('')
# ax2.set_xlabel('')
# ax2.set_ylim(0.375, 0.6)
# ax[0].spines['top'].set_visible(False)
# ax2.spines.top.set_visible(False)
# ax2.spines['bottom'].set_visible(False)
# ax2.spines.bottom.set_linewidth(0)  
# ax[0].set_ylim(0.0, 0.3)
# ax2.tick_params(bottom=False, labelbottom=False)
# ax[0].tick_params(left=False, labelleft=False)


# d = .5  # proportion of vertical to horizontal extent of the slanted line
# kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
#               linestyle="none", color='k', mec='k', mew=1, clip_on=False)
# ax[0].plot([0], [1.], transform=ax[0].transAxes, **kwargs)
# ax2.plot([0], [0], transform=ax2.transAxes, **kwargs)


divider = make_axes_locatable(ax[0])
ax2 = divider.new_horizontal(size="600%", pad=0.1)
fig.add_axes(ax2)


divider2 = make_axes_locatable(ax[1])
ax3 = divider2.new_horizontal(size="600%", pad=0.1)
fig.add_axes(ax3)

sns.lineplot(x="Step", y="minigrid_figure_5-walls-handpicked-reward-model_env-2-1235-1 - rollout/ep_rew_mean_extrinsic", 
			 data=ep_rew_mean_extr[80:], label='', ax=ax3, color='black')

sns.lineplot(x="Step", y="minigrid_figure_5-walls-handpicked-reward-model_env-2-1235-1 - rollout/ep_rew_mean", 
			 data=ep_rew_mean[80:], label='', ax=ax2, color='black')
ax2.set_ylabel('')
ax2.set_xlabel('Episodes (x $10^3$)', size=16)
# ax2.set_ylim(0.375, 0.6)
ax2.set_xlim(71.20000007420757, 176.7999999964663)
# ax[0].spines['right'].set_visible(False)
# ax2.spines.top.set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines.left.set_linewidth(0)
# ax2.spines.bottom.set_linewidth(0)  
ax[0].set_ylim(0.4, 0.6)
ax2.set_xticks([100, 150])
ax2.set_xticklabels([10, 15])
ax2.tick_params(left=False, labelleft=False)
ax[0].tick_params(bottom=False, labelbottom=False)

ax3.set_ylabel('')
ax3.set_xlabel('Episodes (x $10^3$)', size=16)
# ax3.set_ylim(0.375, 0.6)
ax3.set_xlim(71.20000007420757, 176.7999999964663)
# ax[0].spines['right'].set_visible(False)
# ax3.spines.top.set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.spines.left.set_linewidth(0)
# ax3.spines.bottom.set_linewidth(0)  
ax[1].set_ylim(0.3, 0.4)
ax3.set_xticks([100, 150])
ax3.set_xticklabels([10, 15])
ax3.tick_params(left=False, labelleft=False)
ax[1].tick_params(bottom=False, labelbottom=False)

# ax[0].set_xlabel('Episodes (x $10^3$)', size=16)
ax[0].set_ylabel('Internal reward', size=16)
# ax[1].set_xlabel('Episodes (x $10^3$)', size=16)
# print(dir(ax[0].xaxis.label._x))
# print(ax[0].xaxis.label._y)
# print(ax[1].xaxis.label._y)
print(ax[0].yaxis.label._x)
print(ax[0].yaxis.label._y)
# print(ax[1].xaxis.get_label_position())
# ax[0].yaxis.set_label_coords(-.35, 3)
# ax[0].yaxis.set_label_coords(-.1, .5)
ax[0].xaxis.set_label_coords(.6, -0.19)
ax[1].xaxis.set_label_coords(.35, -0.19)
ax[1].set_ylabel('Goals reached', size=16)
ax[0].yaxis.set_ticks([0.4, 0.5, 0.6])
ax[0].yaxis.set_ticklabels([0.4, 0.5, 0.6])
ax[1].yaxis.set_ticks([0.3, 0.35, 0.4])
ax[1].yaxis.set_ticklabels([0.75, 0.875, 1])
# ax[0].set_xlim([80., 180.])
# print(ax[1].get_xlim())
# ax[0].set_xlim(ax[1].get_xlim())
# ax[0].xaxis.set_ticks([100, 150])
ax[1].xaxis.set_ticks([100, 150])
# ax[0].xaxis.set_ticklabels([10, 15])
ax[1].xaxis.set_ticklabels([10, 15])



d = 2.  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax[0].plot([1], [0], transform=ax[0].transAxes, **kwargs)
ax2.plot([0], [0], transform=ax2.transAxes, **kwargs)

ax[1].plot([1], [0], transform=ax[1].transAxes, **kwargs)
ax3.plot([0], [0], transform=ax3.transAxes, **kwargs)


# ax.set_titles("{col_name}")
# # ax.legend.set_title(None)
# ax.legend.set_visible(False)
# # print(dir(ax.legend))
# ax.fig.suptitle('Handpicked Generalizations', y=1.05)
# fig.suptitle("Reward hacking")


# x = ep_rew_mean['Step']
# ymin = ep_rew_mean['minigrid_figure_3a - log_reward_model/Loss/train__MIN']
# ymax = ep_rew_mean['minigrid_figure_3a - log_reward_model/Loss/train__MAX']
# ax.fill_between(x, ymin, ymax, color='blue', alpha=0.25)
# ax.xaxis.set_ticklabels([0, 2, 4, 8, 10, 12])
# ax.xaxis.set_ticks([0, 30, 60, 90, 120])
# ax.xaxis.set_ticklabels([0, 0.1, 0.5, 2])
# ax.xaxis.set_ticks([0, 0.1, 0.5, 2])
# ax.xaxis.set_ticks([60, 90, 120, 150])
# ax.set(xlabel='Environment', ylabel='Reward')
# plt.yscale("log")
# ax.yaxis.set_ticks([5e-4, 5.4e-4, 5.8e-4])
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
plt.subplots_adjust(left=0., bottom=0., top=1., right=1., wspace=.5)
# fig.tight_layout()
sns.despine()#left=True)#right = True, left = True)
# pp.savefig(pad_inches=1.5, bbox_inches='tight')
pp.savefig(bbox_inches='tight')
pp.close()
