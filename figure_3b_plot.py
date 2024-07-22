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

pp = PdfPages('figure_3b-final.pdf')

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['font.family'] = 'monospace'

# ep_rew_mean = pd.read_csv('wandb_export_2023-12-05T15 42 51.403-05 00.csv')

# keys = [
# 	'minigrid_figure_3a - log_reward_model/Loss/train',
# 	'minigrid_figure_3a - log_reward_model/Loss/train__MIN',
# 	'minigrid_figure_3a - log_reward_model/Loss/train__MAX',
# ]

# def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
#     last = scalars[0]  # First value in the plot (first timestep)
#     smoothed = list()
#     for point in scalars:
#         smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
#         smoothed.append(smoothed_val)                        # Save it
#         last = smoothed_val                                  # Anchor the last smoothed value
        
#     return smoothed

# ep_rew_mean[keys[0]] = smooth(ep_rew_mean[keys[0]], 0.8)
# ep_rew_mean[keys[1]] = smooth(ep_rew_mean[keys[1]], 0.8)
# ep_rew_mean[keys[2]] = smooth(ep_rew_mean[keys[2]], 0.8)

# ep_rew_mean['max'] = 0.73
# ep_rew_mean['zero'] = 0.
# ax = sns.lineplot(x="Step", y="max",
# 				  data=ep_rew_mean, color='pink', linestyle='dashed', label='Optimal')
# ax = sns.lineplot(x="Step", y="zero",
# 				  data=ep_rew_mean, color='gray', linestyle='solid')

# data = pd.DataFrame({'x': [4000,1000,500,250,125], 'y': [0.95,0.94,0.81,0.67,0.44]})
# data = pd.DataFrame({'x': [2048,4096,6144,8192,10240], 'y': [0.9826388889,0.9777777778,0.9921296296,0.9909722222,0.9886574074]})
# data = pd.DataFrame({'x': [1e4,2e4,3e4,4e4,8e4,1.2e5], 'y': [0.0005810811537,0.0004167066155,0.0001714363735,0.0000251366272719132,0.0000197858992451264,0.00000606992108431233]})
# data = pd.DataFrame({'x': [1e4,2e4,3e4,4e4,8e4,1.2e5], 
# 'y1': [0.0005810811537,0.0004167066155,0.0001714363735,0.0000251366272719132,0.0000197858992451264,0.00000606992108431233],
# 'y2': [0.000654411411503712,0.0000747795254739116,0.0000561867991212881,np.nan,np.nan,np.nan],
# 'y3': [0.000918567870671229,0.000248195226892964,0.0003297933183240183,0.00020196076468876505,np.nan,np.nan]
# })
# # data['y'] = (data['y1'] + data['y2'] + data['y3'])/3
# data['y'] = data[['y1','y2','y3']].mean(1)
# data['yerr'] = data[['y1','y2','y3']].std(1)
# print(data['y'])
# print(data['yerr'])

data = pd.DataFrame({'x': [
# 1e4,2e4,3e4,4e4,8e4,1.2e5,
# 1e4,2e4,3e4,4e4,8e4,1.2e5,
# 1e4,2e4,3e4,4e4,8e4,1.2e5,
9216, 18432, 27648, 36864, 73728, 108748,
9216, 18432, 27648, 36864, 73728, 108748,
9216, 18432, 27648, 36864, 73728, 108748,

9216, 18432, 27648, 36864, 73728, 108748,
9216, 18432, 27648, 36864, 73728, 108748,
9216, 18432, 27648, 36864, 73728, 108748,
9216, 18432, 27648, 36864, 73728, 108748,
9216, 18432, 27648, 36864, 73728, 108748,
9216, 18432, 27648, 36864, 73728, 108748,
9216, 18432, 27648, 36864, 73728, 108748,
],
'y': [
    # 0.0024964132853710908,0.0010721962348725356,0.0003414916487827213,0.00014747823966843043,2.3173092850603538e-05,8.045912340762867e-06,
    # 0.0010714711257833187,0.0005739589009863266,0.0002293582987059987,0.00019649392527298915,4.360819415794026e-05,1.0982323651086894e-05,
    # 0.002809879882988753,0.0012097695133661546,0.0004080900452739418,0.00019816647142789634,3.8426860025531084e-05,1.1288348041743765e-05,
    # # 0.0005810811537,0.0004167066155,0.0001714363735,0.0000251366272719132,0.0000197858992451264,0.00000606992108431233,
    # # 0.000654411411503712,0.0000747795254739116,0.0000561867991212881,
    # # 0.000918567870671229,0.000248195226892964,0.0003297933183240183,0.00020196076468876505,8.989303576994929e-07,5.852301005942332e-07,

    0.001046909826, 0.0007351661462, 0.0005810785056330958, 0.00035387944346218676, 0.00000800842055201069, 0.00000251770828543235,
    0.001287588096,  0.001240031379,  0.0005333652474, 0.0003203592087, 0.000020241580880837, 0.00000279865967459237,
    0.00161235415,   0.0004214657977, 0.0001190424156, 0.0001133762415, 0.000007576699026, 0.00000359247854,

    0.002073635471753265,0.0013159025231144637,0.0006637037826996003,0.0003131024617963518,1.040598724298894e-05,5.071390788670502e-06,
    0.00043366326356095934,9.587284971687211e-05,5.02656921868376e-05,4.1969158817621784e-05,7.945781554508492e-06,2.5452746350657408e-06,
    0.0015372747187400921,0.0013834323919969253,0.0007123450773976778,0.00039788929902339574,1.9237686678260483e-05,7.0534733859303105e-06,
    0.0012295442151284489,0.0009000607758644717,0.0003452798543519215,6.075622501462412e-05,6.42837787452221e-06,3.6837034302791307e-06,
    0.0014885213729390308,0.0012332979327769724,0.000818451246899411,0.0005585317807277991,4.7920886782146245e-05,1.3362667798278295e-05,
    0.0009952264773659342,0.0010545298498810802,0.0006908453713018494,0.0004882629579507248,6.009844069623554e-05,1.130437510726482e-05,
    0.0007765436986561687,0.00031857541257254593,0.0002901249805889175,0.00016970844888472125,1.7534363765948097e-05,2.8379202439975497e-06,

],
})

data['x'] = data["x"].astype('int')

# import pdb; pdb.set_trace()

ax = sns.pointplot(x="x", y="y", native_scale=True,# markers="o", 
                  data=data, color='blue', label='')
# ax = sns.lineplot(x="x", y="y", 
# 				  data=data, alpha=0.5, color='blue', label='')
# ax = sns.lineplot(x="x", y="y2", 
#                   data=data, alpha=0.5, color='blue', label='')
# ax.axvline(x = 4, ymin = 0, ymax = 1, color='black')
# plt.legend()
# plt.title("More data improves performance")
# plt.title("Training reward model on more\n data improves performance")


# x = ep_rew_mean['Step']
# ymin = ep_rew_mean['minigrid_figure_3a - log_reward_model/Loss/train__MIN']
# ymax = ep_rew_mean['minigrid_figure_3a - log_reward_model/Loss/train__MAX']
# ax.fill_between(x, ymin, ymax, color='blue', alpha=0.25)
# print(ax.get_xlim())
ax.xaxis.set_ticklabels([0, 4, 8, 12])
ax.xaxis.set_ticks([0, 4e4, 8e4, 12e4,])
# plt.xticks([1e4, 2e4, 12e4])
# ax.xaxis.set_ticklabels([0, 2, 4, 8, 10, 12])
# ax.xaxis.set_ticks([0, 30, 60, 90, 120])
# ax.xaxis.set_ticks([0, 1, 2, 3, 4, 5])
# print(dir(ax.xaxis))
# ax.set_xscale('linear')
# ax.set(xlabel='RL timesteps (x $10^4$)', ylabel='MSE Loss')
ax.set(xlabel='Social rewards (x $10^4$)', ylabel='MSE Loss')
plt.yscale("log")
# ax.set_ylim(0, 1)
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
