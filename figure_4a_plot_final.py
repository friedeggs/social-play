from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

import numpy as np

import pandas as pd
import seaborn as sns
sns.set_theme(style="white")
sns.set_context("paper", font_scale=2.5)

pp = PdfPages('figure_4a-final.pdf')
plt.rcParams['font.family'] = 'monospace'

# plt.rcParams['figure.constrained_layout.use'] = True

# ep_rew_mean = pd.read_csv('wandb_export_2023-12-04T22 59 46.412-05 00.csv')
# ep_rew_mean = pd.read_csv('wandb_export_2023-12-05T16 00 15.669-05 00.csv')
# ep_rew_mean = pd.read_csv('wandb_export_2023-12-05T22 57 40.237-05 00.csv')
# ep_rew_mean = pd.read_csv('wandb_export_2023-12-06T04 04 51.279-05 00.csv')
# ep_rew_mean = pd.read_csv('wandb_export_2023-12-06T17 18 46.267-05 00.csv')
# ep_rew_mean = pd.read_csv('wandb_export_2024-01-21T23 33 09.287-05 00.csv')
ep_rew_mean = pd.read_csv('wandb_export_2024-03-08T19 53 06.373-05 00.csv')

# print(ep_rew_mean.keys())


keys = [
	'minigrid_figure_1b-none - rollout/ep_rew_mean_extrinsic',
	'minigrid_figure_1b-none - rollout/ep_rew_mean_extrinsic__MIN',
	'minigrid_figure_1b-none - rollout/ep_rew_mean_extrinsic__MAX',
	'minigrid_figure_1b-oracle - rollout/ep_rew_mean_extrinsic',
	'minigrid_figure_1b-oracle - rollout/ep_rew_mean_extrinsic__MIN',
	'minigrid_figure_1b-oracle - rollout/ep_rew_mean_extrinsic__MAX'
	'minigrid_figure_1b-reward-model - rollout/ep_rew_mean_extrinsic',
	'minigrid_figure_1b-reward-model - rollout/ep_rew_mean_extrinsic__MIN',
	'minigrid_figure_1b-reward-model - rollout/ep_rew_mean_extrinsic__MAX'
]

for name in ['none', 'oracle', 'reward-model']:
	columns = []
	for seed in range(1234,1244):
		key = f'minigrid_figure_1b-{name}-{seed} - rollout/ep_rew_mean_extrinsic'
		columns.append(key)
		ep_rew_mean[key][0] *= 9/20
		ep_rew_mean[key][1] *= 19/20
		assert key in ep_rew_mean, key
	ep_rew_mean[f'minigrid_figure_1b-{name} - rollout/ep_rew_mean_extrinsic'] = ep_rew_mean[columns].mean(1)
	ep_rew_mean[f'minigrid_figure_1b-{name} - rollout/ep_rew_mean_extrinsic__MIN'] = ep_rew_mean[columns].min(1)
	ep_rew_mean[f'minigrid_figure_1b-{name} - rollout/ep_rew_mean_extrinsic__MAX'] = ep_rew_mean[columns].max(1)
	print(ep_rew_mean[f'minigrid_figure_1b-{name} - rollout/ep_rew_mean_extrinsic'])
	# print(ep_rew_mean[columns].sum() / 5)
	# print(ep_rew_mean[columns].mean(1))

ep_rew_mean_half = ep_rew_mean[:60]
ep_rew_mean_other_half = ep_rew_mean[60:]
max_y = ep_rew_mean_other_half['minigrid_figure_1b-oracle - rollout/ep_rew_mean_extrinsic'].max()
print(max_y)
# ep_rew_mean['max'] = 0.73
# ep_rew_mean['zero'] = 0.
# ax = sns.lineplot(x="Step", y="max",
# 				  data=ep_rew_mean, color='pink', linestyle='dashed', label='Optimal')
# ax = sns.lineplot(x="Step", y="zero",
# 				  data=ep_rew_mean, color='gray', linestyle='solid')
# ax_de = sns.lineplot(x="Step", y="minigrid_figure_1b-oracle - rollout/ep_rew_mean_extrinsic", 
# 				  data=ep_rew_mean, alpha=0.5, color='blue', label='With caregiver')
ax_de = sns.lineplot(x="Step", y="minigrid_figure_1b-oracle - rollout/ep_rew_mean_extrinsic", 
				  data=ep_rew_mean_half, alpha=0.5, color='blue', label='With social rewards', linewidth=3)
ax_rm = sns.lineplot(x="Step", y="minigrid_figure_1b-reward-model - rollout/ep_rew_mean_extrinsic", 
				  data=ep_rew_mean_other_half, alpha=0.5, color='red', label='With ISR model', linewidth=3)
ax_nr = sns.lineplot(x="Step", y="minigrid_figure_1b-none - rollout/ep_rew_mean_extrinsic",
             data=ep_rew_mean_other_half, alpha=0.5, color='green', label='Without ISR model', linewidth=3)
# ax_nr.axhline(y = max_y, xmin = 0, xmax = 1, color='blue', linestyle="dashed") # 0.515
ax_nr.axvline(x = 60, ymin = 0, ymax = 1, color='black', linestyle="dashed")
# plt.legend()
# plt.legend(loc=[.53,.82], prop={'size': 12})
# plt.legend(loc=[.53,.82], prop={'size': 12})
plt.legend(loc=[.05,.82], prop={'size': 18})
# plt.title("Learning vs. Caregiver Presence")
# plt.title("Learning with Internalized Reward")
# plt.title("Learning with internalized social rewards", y=1.08)
# plt.title("Learning with internalized social rewards", y=1.35)
# plt.title("Learning with internalized social rewards", y=1.15)



x = ep_rew_mean_other_half['Step']
ymin = ep_rew_mean_other_half['minigrid_figure_1b-none - rollout/ep_rew_mean_extrinsic__MIN']
ymax = ep_rew_mean_other_half['minigrid_figure_1b-none - rollout/ep_rew_mean_extrinsic__MAX']
ax_nr.fill_between(x, ymin, ymax, color='green', alpha=0.25)
# x = ep_rew_mean['Step']
# ymin = ep_rew_mean['minigrid_figure_1b-none - rollout/ep_rew_mean_extrinsic__MIN']
# ymax = ep_rew_mean['minigrid_figure_1b-none - rollout/ep_rew_mean_extrinsic__MAX']
# ax_nr.fill_between(x, ymin, ymax, color='green', alpha=0.25)
x = ep_rew_mean_half['Step']
ymin = ep_rew_mean_half['minigrid_figure_1b-oracle - rollout/ep_rew_mean_extrinsic__MIN']
ymax = ep_rew_mean_half['minigrid_figure_1b-oracle - rollout/ep_rew_mean_extrinsic__MAX']
ax_de.fill_between(x, ymin, ymax, color='blue', alpha=0.25)
x = ep_rew_mean_other_half['Step']
ymin = ep_rew_mean_other_half['minigrid_figure_1b-reward-model - rollout/ep_rew_mean_extrinsic__MIN']
ymax = ep_rew_mean_other_half['minigrid_figure_1b-reward-model - rollout/ep_rew_mean_extrinsic__MAX']
ax_rm.fill_between(x, ymin, ymax, color='red', alpha=0.25)
ax_nr.set(xlabel='Episodes (x $10^3$)', ylabel='Goals reached')

# ax_nr.xaxis.set_ticklabels([0, int(4e4//20), int(8e4//20), int(12e4//20), int(16e4//20), int(20e4//20), int(24e4//20)])
# 0, 3, 6, 9, 12
# 2
# 20
# ax_nr.xaxis.set_ticks([0, 20, 40, 60, 80, 100, 120])
ax_nr.xaxis.set_ticklabels([0, 3, 6, 9, 12])
ax_nr.xaxis.set_ticks([0, 30, 60, 90, 120])
# ax_nr.yaxis.set_ticks([0, 0.4, 0.8, 1.2, 1.6])
ax_nr.yaxis.set_ticklabels([0, 0.2/0.4, 0.4/0.4, 1.5, 0.8/0.4])

plt.ylim(0, 0.8)
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
# ax_de.set_aspect('equal', adjustable='box')
# ax_nr.set_aspect('equal', adjustable='box')
sns.despine()# right = True, left = True)
plt.tight_layout()
# plt.savefig('figure_1b.png')
pp.savefig(bbox_inches='tight')

pp.close()