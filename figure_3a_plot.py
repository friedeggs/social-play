from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

import numpy as np

from typing import List

import pandas as pd
import seaborn as sns
sns.set_theme(style="white")
sns.set_context("paper", font_scale=3)

pp = PdfPages('figure_3a-final.pdf')

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['font.family'] = 'monospace'

# loss_train = pd.read_csv('wandb_export_2023-12-05T15 42 51.403-05 00.csv')
# loss_train = pd.read_csv('wandb_export_2023-12-06T23 12 16.222-05 00.csv')
# loss_train = pd.read_csv('wandb_export_2023-12-07T00 07 10.423-05 00.csv')
# loss_test = pd.read_csv('wandb_export_2023-12-07T00 07 30.891-05 00.csv')
# loss_train = pd.read_csv('wandb_export_2023-12-07T17 06 28.184-05 00.csv')
# loss_test = pd.read_csv('wandb_export_2023-12-07T17 06 45.473-05 00.csv')
# loss_train = pd.read_csv('wandb_export_2023-12-12T05 05 01.223-05 00.csv')
# loss_test = pd.read_csv('wandb_export_2023-12-12T05 05 25.814-05 00.csv')
loss_train = pd.read_csv('wandb_export_2024-01-29T02 35 58.698-05 00.csv')
loss_test = pd.read_csv('wandb_export_2024-01-29T02 36 23.350-05 00.csv')

# loss_train = loss_train[:12081]
# loss_test = loss_test[:12081]
loss_train = loss_train[:5000]
loss_test = loss_test[:5000]

# print(loss_train.keys())

keys = [
	'minigrid_figure_3a - log_reward_model/Loss/train',
	'minigrid_figure_3a - log_reward_model/Loss/train__MIN',
	'minigrid_figure_3a - log_reward_model/Loss/train__MAX',
]
# print(loss_test.keys())

def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

loss_train[keys[0]] = smooth(loss_train[keys[0]], 0.8)
loss_train[keys[1]] = smooth(loss_train[keys[1]], 0.8)
loss_train[keys[2]] = smooth(loss_train[keys[2]], 0.8)


keys = [
	'minigrid_figure_3a - log_reward_model/Loss/test',
	'minigrid_figure_3a - log_reward_model/Loss/test__MIN',
	'minigrid_figure_3a - log_reward_model/Loss/test__MAX',
]

loss_test[keys[0]] = smooth(loss_test[keys[0]], 0.8)
loss_test[keys[1]] = smooth(loss_test[keys[1]], 0.8)
loss_test[keys[2]] = smooth(loss_test[keys[2]], 0.8)

# loss_train['max'] = 0.73
# loss_train['zero'] = 0.
# ax = sns.lineplot(x="Step", y="max",
# 				  data=loss_train, color='pink', linestyle='dashed', label='Optimal')
# ax = sns.lineplot(x="Step", y="zero",
# 				  data=loss_train, color='gray', linestyle='solid')
ax = sns.lineplot(x="Step", y="minigrid_figure_3a - log_reward_model/Loss/train", 
				  data=loss_train, alpha=0.5, color='blue', label='Train')
ax = sns.lineplot(x="Step", y="minigrid_figure_3a - log_reward_model/Loss/test", 
				  data=loss_test, alpha=0.5, color='red', label='Test')
# ax.axvline(x = 4, ymin = 0, ymax = 1, color='black')
plt.legend()
# plt.title("Reward model training")


# x = loss_test['Step']
# ymin = loss_test['minigrid_figure_3a - log_reward_model/Loss/test__MIN']
# ymax = loss_test['minigrid_figure_3a - log_reward_model/Loss/test__MAX']
# ax.fill_between(x, ymin, ymax, color='blue', alpha=0.25)
# x = loss_train['Step']
# ymin = loss_train['minigrid_figure_3a - log_reward_model/Loss/train__MIN']
# ymax = loss_train['minigrid_figure_3a - log_reward_model/Loss/train__MAX']
# ax.fill_between(x, ymin, ymax, color='blue', alpha=0.25)
ax.set(xlabel='Social rewards (x $10^3)$', ylabel='MSE Loss')

ax.xaxis.set_label_coords(0.45, -.15)

print(ax.xaxis.label._x)
print(ax.xaxis.label._y)

# ax.xaxis.set_ticklabels([0, 1, 2, 3, 4])
# ax.xaxis.set_ticks([0, 10000, 20000, 30000, 40000])

# ax.xaxis.set_ticklabels([0, 1, 2, 3, 4, 5])
# ax.xaxis.set_ticks([0, 1000, 2000, 3000, 4000])

ax.xaxis.set_ticklabels([0, 2, 4])
ax.xaxis.set_ticks([0, 2000, 4000])

# ax.xaxis.set_ticklabels([0, 4, 8, 12])
# ax.xaxis.set_ticks([0, 4000, 8000, 12000])
# ax.yaxis.set_ticks([0, 0.4, 0.8, 1.2, 1.6])
# ax.yaxis.set_ticks([0, 0.05, 0.1, 0.15])
ax.yaxis.set_ticks([0, 0.05, 0.1])

# plt.axis('scaled')
# plt.axis('square')
# plt.gca().set_aspect(2)
# plt.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0,
#                             wspace=0)
# plt.tight_layout()
# plt.tight_layout(pad=0, w_pad=0)#, h_pad=0)
# plt.gca().autoscale_view('tight')
# plt.axis('equal')
# plt.gca().set_aspect('equal', adjustable='box')
# ax.set_aspect('equal', adjustable='box')
# ax_nr.set_aspect('equal', adjustable='box')
sns.despine()#right = True, left = True)
pp.savefig()

pp.close()
