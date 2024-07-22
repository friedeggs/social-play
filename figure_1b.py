from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

# from minigrid.wrappers import ImgObsWrapper, NoDeath, RGBImgPartialObsWrapper
from stable_baselines3 import PPO

from typing import Any, Iterable, SupportsFloat, TypeVar
from typing import Callable, Dict, List, Optional, Tuple, Union

from tqdm import tqdm


import gymnasium as gym
from gymnasium.core import ActType, ObsType, WrapperObsType

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import matplotlib
import matplotlib.pyplot as plt

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import os

import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback


from gymnasium import spaces
from gymnasium.core import ActionWrapper, ObservationWrapper, ObsType, Wrapper
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX


import warnings


from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped, VecTransposeImage

import time
import wandb

import argparse
from distutils.util import strtobool

# 1235, 1237, 1238, 1239, 1240
# seed = 1236
# seed = 1241
# seed = 1242

# seed = 1234
# seed = 1235
# seed = 1236
# seed = 1237
# seed = 1238
np.set_printoptions(precision=2, suppress=True)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='reward-model', # options: ['reward-model', 'oracle', 'none']
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1234,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="social-play",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="",
        help="the entity (team) of wandb's project")
    args = parser.parse_args()
    # fmt: on
    return args

def encode_observation(env):
    env = env.unwrapped
    full_grid = env.grid.encode()
    full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir]
        )
    w = env.unwrapped.width-1-1
    h = env.unwrapped.height-1-1
    full_grid = np.pad(full_grid, ((w, h), (w, h), (0, 0)), 'constant', constant_values=0)
    partial_grid = full_grid[env.agent_pos[0]+1+w-w:env.agent_pos[0]+w+w,env.agent_pos[1]+1+h-h:env.agent_pos[1]+h+h]
    for i in range(env.agent_dir):
        partial_grid = np.rot90(partial_grid, k=1, axes=(1,0))
    partial_grid = np.rot90(partial_grid, k=1, axes=(1,0))
    partial_grid = np.rot90(partial_grid, k=1, axes=(1,0))
    partial_grid = np.flip(partial_grid, axis=1).copy()
    is_goal = np.expand_dims(partial_grid[...,0] == OBJECT_TO_IDX["goal"], -1)
    partial_grid = np.concatenate([partial_grid, is_goal], axis=-1)
    return partial_grid

class FullyObsWrapper(ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding instead of the agent view.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import FullyObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> obs, _ = env.reset()
        >>> obs['image'].shape
        (7, 7, 3)
        >>> env_obs = FullyObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> obs['image'].shape
        (11, 11, 3)
    """

    def __init__(self, env):
        super().__init__(env)

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width*2-5, self.env.height*2-5, 4),  # number of cells
            dtype="uint8",
        )

        self.observation_space = new_image_space

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir]
        )
        w = self.env.unwrapped.width-1-1
        h = self.env.unwrapped.height-1-1
        # print(w, h) w + h - 1  self.env.width-2 2 * size - 5

        # # print(full_grid.shape)
        # is_target = np.zeros((full_grid.shape[0], full_grid.shape[1], 1))
        # is_target[self.target[0], self.target[1]] = 1
        # full_grid = np.concatenate([full_grid, is_target], axis=-1)

        full_grid = np.pad(full_grid, ((w, h), (w, h), (0, 0)), 'constant', constant_values=0)
        # print(full_grid.shape)
        # print(env.agent_pos)
        # print(w, h)
        # print(env.agent_pos[0]+w-w,env.agent_pos[0]+w+w)
        # print(env.agent_pos[1]+h-h,env.agent_pos[1]+h+h)

        # import pdb; pdb.set_trace()
        # print(self.target)
        partial_grid = full_grid[env.agent_pos[0]+1+w-w:env.agent_pos[0]+w+w,env.agent_pos[1]+1+h-h:env.agent_pos[1]+h+h]
        for i in range(env.agent_dir):
            partial_grid = np.rot90(partial_grid, k=1, axes=(1,0))
        partial_grid = np.rot90(partial_grid, k=1, axes=(1,0))
        partial_grid = np.rot90(partial_grid, k=1, axes=(1,0))
        partial_grid = np.flip(partial_grid, axis=1).copy()
        # print(partial_grid.shape)

        # # partial_grid[partial_grid[...,0] == OBJECT_TO_IDX["lava"],-1] = 1
        # is_lava = np.expand_dims(partial_grid[...,0] == OBJECT_TO_IDX["lava"], -1)
        # partial_grid = np.concatenate([partial_grid, is_lava], axis=-1)
        is_goal = np.expand_dims(partial_grid[...,0] == OBJECT_TO_IDX["goal"], -1)
        partial_grid = np.concatenate([partial_grid, is_goal], axis=-1)
        # is_term = (partial_grid[...,0] == OBJECT_TO_IDX["lava"]) | (partial_grid[...,0] == OBJECT_TO_IDX["goal"])
        # is_term = is_term & ~partial_grid[...,-1].astype(bool)
        # is_term = np.expand_dims(is_term, -1)
        # partial_grid = np.concatenate([partial_grid, is_term], axis=-1)
        # # is_target = np.zeros_like(is_lava)
        # # is_target = np.zeros((partial_grid.shape[0], partial_grid.shape[1], 1))
        # # is_target[self.target[0], self.target[1]] = 1
        # # partial_grid = np.concatenate([partial_grid, is_target], axis=-1)
        # # import pdb; pdb.set_trace()
        return partial_grid

    def step(
        self, action: ActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.observation` on the returned observations."""
        obs = self.env.gen_obs()
        self.env.obs = self.observation(obs)

        observation, reward, terminated, truncated, info = self.env.step(action)

        obs = self.observation(observation)
        # self.env.obs = obs

        return obs, reward, terminated, truncated, info

def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed


class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        ## Reward model
        observation_space = spaces.Box(
            low=0,
            high=255,
            # shape=(13, 13, 5),  # number of cells
            shape=(9, 9, 5),  # number of cells
            dtype="uint8",
        )
        n_input_channels = observation_space.shape[0]
        features_dim = 1
        self.cnn = nn.Sequential(
            # nn.BatchNorm2d(n_input_channels),
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            # nn.Conv2d(32, 64, (2, 2)),
            # nn.ReLU(),
            # nn.Flatten(0, -1),
            nn.Flatten(1, -1),
        )
        # self.flatten = nn.Flatten(0, -1)
        # self.flatten = nn.Flatten(1, -1)
        with torch.no_grad():
            # n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[0]
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
            return layer

        self.linear = nn.Sequential(
            layer_init(nn.Linear(n_flatten, 32)),
            nn.ReLU(), 
            layer_init(nn.Linear(32, 32)),
            nn.ReLU(), 
            # layer_init(nn.Linear(32, 32)),
            # nn.ReLU(), 
            # layer_init(nn.Linear(32, 32)),
            # nn.ReLU(), 
            layer_init(nn.Linear(32, features_dim)),
            # nn.ReLU(), 
            # nn.Softmax(-1)
            # nn.Sigmoid(),
        )

        # self.rewardmodel = nn.Sequential(
        #     # nn.Flatten(),
        #     nn.Linear((2*self.width-3) * (2*self.height-3) * 3, 1),
        #     nn.ReLU(),
        #     # nn.Softmax(dim=-1),
        # )
        # self.loss_fn = nn.BCELoss()
        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(list(self.cnn.parameters()) + list(self.linear.parameters()), lr=3e-4, eps=1e-5)
        # self.optimizer = optim.Adam(self.linear.parameters(), lr=3e-4, eps=1e-5)
        self.logs = []
        self.data = []
        self.num_reward = 0
        self.train = True

    def encode_action(self, state, action):
        action_enc = np.expand_dims(state[...,0] == OBJECT_TO_IDX["agent"], -1)
        action_enc = action_enc.astype(np.float32)
        action_enc *= action
        state_action = np.concatenate([state, action_enc], axis=-1)
        return state_action

    def observe(self, state, action, reward):
        if reward > 0.:
            self.num_reward += 1
            state_action = self.encode_action(state, action)
            self.data.append((state_action, action, reward))
            # state_action = self.encode_action(state, 0)
            # self.data.append((state_action, 0, 0))
            # state_action = self.encode_action(state, 1)
            # self.data.append((state_action, 1, 0))
        else:
            state_action = self.encode_action(state, action)
            self.data.append((state_action, action, reward))

    def learn(self, n_steps=1e4):
        # num_reward = self.num_reward
        self.train_data = self.data[:len(self.data)*9//10]
        self.test_data = self.data[len(self.data)*9//10:]
        num_reward = sum([1 if reward > 0. else 0 for _, _, reward in self.train_data])
        # num_reward = sum([1 if reward > 0. else 0 for _, _, reward in self.train_data][:len(self.train_data)//2])
        # num_reward = sum([1 if reward > 0. else 0 for _, _, reward in self.train_data][len(self.train_data)//2:])
        num_data = len(self.train_data)
        # num_data = len(self.train_data)//2
        frac_rew = 1./(2*num_reward)
        frac = 1./(2*(num_data-num_reward))
        train_weights = [frac_rew if reward > 0. else frac for _, _, reward in self.train_data]
        # weights = [frac_rew if reward > 0. else frac for _, _, reward in self.train_data][:len(self.train_data)//2]
        # weights = [frac_rew if reward > 0. else frac for _, _, reward in self.train_data][len(self.train_data)//2:]
        # losses = []

        num_reward = sum([1 if reward > 0. else 0 for _, _, reward in self.test_data])
        num_data = len(self.test_data)
        frac_rew = 1./(2*num_reward)
        frac = 1./(2*(num_data-num_reward))
        test_weights = [frac_rew if reward > 0. else frac for _, _, reward in self.test_data]

        num_epochs = 1
        batch_size = 1
        # writer = torch.utils.tensorboard.SummaryWriter(log_dir="log_reward_model")
        train_sampler = torch.utils.data.WeightedRandomSampler(train_weights, num_samples=len(self.train_data), replacement=True)
        train_dataloader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, sampler=train_sampler, drop_last=True)
        test_sampler = torch.utils.data.WeightedRandomSampler(test_weights, num_samples=len(self.test_data), replacement=True)
        test_dataloader = torch.utils.data.DataLoader(self.test_data, batch_size=batch_size, sampler=test_sampler, drop_last=True)
        # for i in tqdm(range(int(n_steps))):
        for epoch_id in range(num_epochs):
            for i, batch in tqdm(enumerate(train_dataloader)):
                # # import pdb; pdb.set_trace()
                # idx = np.random.choice(np.arange(len(self.data)), p=weights)#; self.data[idx][2]
                # # idx = np.random.choice(np.arange(len(self.data)//2), p=weights)#; self.data[idx][2]
                # # idx = np.random.choice(np.arange(len(self.data)//2, len(self.data)), p=weights)#; self.data[idx][2]
                # state_action, action, reward = self.data[idx]
                state_action, action, reward = batch
                # state_action = state_action.reshape(batch_size, -1)
                # reward = reward.reshape(batch_size, -1)
                pred, loss = self.forward(state_action, reward)

                # writer.add_scalar('Loss/train', loss.detach(), epoch_id * len(train_dataloader) + i)

                # loss_tot = 0.
                # for j, batch in enumerate(test_dataloader):
                #     state_action, action, reward = batch
                #     pred, loss = self.forward(state_action, reward)
                #     loss_tot += loss
                #     if j >= 10:
                #         break
                # loss_tot /= len(test_dataloader)
                # writer.add_scalar('Loss/test', loss_tot.detach(), epoch_id * len(train_dataloader) + i)
        #     losses.append(loss.detach())
        # losses = smooth(losses, 0.8)
        # ymin, ymax = plt.gca().get_ylim()
        # plt.plot(range(len(losses)), losses, 'o-')
        # plt.axvline(x = 1e4, ymin = ymin, ymax = ymax)
        # plt.axvline(x = 2e4, ymin = ymin, ymax = ymax)
        # plt.savefig('training_curve.png')

    def evaluate(self, env, policy, n_eval_episodes=10):
        acc = 0.
        count = 0.

        for i in range(int(n_eval_episodes)):

            obs = env.reset()
            # state = encode_observation(env)
            # state = encode_observation(base_env)
            # action = env.action_space.sample()
            # action, _ = policy.predict(obs)
            # state_action = self.encode_action(state, action)
            # pred = self.forward(state_action)
            # assume False
            # import pdb; pdb.set_trace()
            # acc += not (pred >= 0.5)

            # print('reset')
            # print(f'{pred.item():0.2f}')

            pred = None
            terminated = False
            truncated = False

            if isinstance(obs, tuple):
                obs = obs[0]
            while not (terminated or truncated):
                if pred is not None:
                    # acc += not (pred >= 0.5)
                    acc += 1 - pred
                    count += 1
                action, _ = policy.predict(obs)
                # state = encode_observation(base_env)
                # import pdb; pdb.set_trace()
                state = encode_observation(env)
                state_action = self.encode_action(state, action)
                pred = self.forward(state_action[None])[0]
                obs, reward, terminated, truncated, info = env.step(action)
            # state_action = self.encode_action(state, action)
            # pred = self.forward(state_action)
            if terminated:
                # print('terminated')
                # import pdb; pdb.set_trace()
                # acc += pred >= 0.5
                acc += pred
                count += 1
            else: # truncated
                # print('truncated')
                # acc += not (pred >= 0.5)
                acc += 1 - pred
                count += 1
            # print(f'{pred.item():0.2f}')

            env.reset()

        # return acc / (n_eval_episodes * 2.)
        return acc / count

    def evaluate_enumerate(self, env_class, log_dir, accuracy, filename):

        matrix = np.zeros((2,2))
        counts = np.zeros((2,2))

        rewards = []
        preds = []

        # for idx in range(100):
        # for goal_idx in [0]: #range(9):
        #     for agent_idx in [1]: #range(9):
        for goal_idx in range(25):
            for agent_idx in range(25):
                if goal_idx == agent_idx: continue
                for agent_dir in range(4):
        #         for agent_dir in range(4):
                    
                    # agent_goal_idx = np.random.choice(72)
                    # goal_idx = agent_goal_idx % 9
                    # agent_idx = agent_goal_idx // 9
                    # agent_dir = np.random.choice(4)
                    goal_pos = (goal_idx % 5 + 1, goal_idx // 5 + 1)

                    # height = 7

                    # if env_class == SimpleEnv:
                    #     if goal_pos[1] >= height//2+1:
                    #         continue

                    # if env_class == ComplexEnv:
                    #     if goal_pos[1] < height//2+1:
                    #         continue

                    agent_pos = (agent_idx % 5 + 1, agent_idx // 5 + 1)

                    eval_env = env_class(
                        n_walls=3,#3 if env_class == ComplexEnv else 1,
                        goal_pos=goal_pos, 
                        agent_start_pos=agent_pos, 
                        agent_start_dir=agent_dir, 
                        render_mode="rgb_array")
                    eval_env = FullyObsWrapper(eval_env)
                    eval_env = Monitor(eval_env, log_dir)
                    
                    for action in [0,1,2]:
                    # for action in [2]:

                        obs = eval_env.reset()
                        if isinstance(obs, tuple):
                            obs = obs[0]

                        state = encode_observation(eval_env)
                        state_action = self.encode_action(state, action)
                        pred = self.forward(state_action[None])[0]
                        pred = pred.item()

                        obs, reward, terminated, truncated, info = eval_env.step(action)
                        reward = max(reward, 0)


                        # print(reward, pred)

                        if int(reward) == 1.:
                            matrix[int(reward)][round(pred)] += pred
                        else:
                            matrix[int(reward)][round(pred)] += 1-pred

                        counts[int(reward)][round(pred)] += 1

                        rewards.append(reward)
                        preds.append(pred)

                        # if int(reward) != round(pred):
                        #     # print(action)

                        #     eval_env = env_class(
                        #         n_walls=0,#3 if env_class == ComplexEnv else 1,
                        #         goal_pos=goal_pos, 
                        #         agent_start_pos=agent_pos, 
                        #         agent_start_dir=agent_dir, 
                        #         render_mode="rgb_array")
                        #     eval_env = FullyObsWrapper(eval_env)
                        #     eval_env = Monitor(eval_env, log_dir)
                        #     obs = eval_env.reset()
                        #     if isinstance(obs, tuple):
                        #         obs = obs[0]
                        #     # import pdb; pdb.set_trace()
                        #     obs, reward, terminated, truncated, info = eval_env.step(action)
                        #     # import pdb; pdb.set_trace()


        np.set_printoptions(precision=2, suppress=True)
        print(matrix / counts)
        print(matrix)
        print(counts)

        fig, ax = plt.subplots()
        ax.scatter(rewards, preds, alpha=0.2)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        plt.xlabel('Reward')
        plt.ylabel('Prediction')
        plt.title(f'On-policy accuracy: {accuracy:2f}')
        fig.savefig(filename)

        return matrix, counts



    def forward(self, obs, gt=None):# action=None, actions=None, prev_dir=None, agent_dir=None, fwd_cell=None):
        ### Reward model
        # observations = torch.from_numpy(obs['image']).float()
        # observations[self.agent_pos[0]][self.agent_pos[1]] = torch.tensor(
        #     [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], self.agent_dir]
        # )
        # import pdb; pdb.set_trace()
        obs = torch.as_tensor(obs).float()
        # pred = self.linear(self.cnn(observations))
        # cur_dir = torch.as_tensor(DIR_TO_VEC[agent_dir])
        # prev_dir = torch.as_tensor(DIR_TO_VEC[prev_dir])
        # # dirs = torch.cat([cur_dir, prev_dir]).float()
        # if fwd_cell is None:
        #     obj = -1
        # else:
        #     obj = OBJECT_TO_IDX[fwd_cell.type]
        # obj = torch.Tensor([obj]).float()
        # dirs = torch.cat([cur_dir, prev_dir, obj]).float()
        # pred = self.linear(dirs)
        pred1 = self.cnn(obs)
        # import pdb; pdb.set_trace()
        # pred2 = self.flatten(pred1)
        pred = self.linear(pred1)
        pred = pred.squeeze(1)
        # pred = self.linear(self.cnn(obs))
        # pred = F.softmax(pred)
        # pred = pred[0][None]

        if gt is not None:
            # if action == actions.forward: 
            #     if fwd_cell is not None and fwd_cell.type == "goal":
            #         gt = 1
            #     else:
            #         # gt = 1
            #         gt = 0
            # else:
            #     gt = 0
            # gt = max(gt, 0)
            # gt = torch.Tensor([gt]).float()
            gt = torch.clip(gt, min=0).float()
            # _gt = 1
            # _gt = torch.Tensor([_gt]).float()

            loss = self.loss_fn(pred, gt)
            # print(self.loss_fn(pred, _gt))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()

            # if loss > 1e-4:
            #     # import pdb; pdb.set_trace()
            #     print(obs[...,0], obs[...,-1], gt)

            return pred, loss
            # self.logs.append([len(self.logs), loss.detach().numpy(), pred.detach().numpy(), gt.numpy()])
            # print(loss, pred, gt)
            # print(f'{loss.item():.2f}\t{pred.item():.2f}\t{gt.item():.2f}')
            # import pdb; pdb.set_trace()
        return pred


class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=7,
        agent_view_size=13,
        agent_start_pos=None, #(1, 1),
        agent_start_dir=None, #0,
        max_steps: int | None = None,
        reward_model=None,
        goal_pos=None, # (3, 3),
        n_walls=0,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = goal_pos
        self.n_walls = n_walls

        self.reset_counter = 0

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * (size-2)**2 // 5 # 20 timesteps

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # agent_view_size=agent_view_size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        # Allow only 3 actions permitted: left, right, forward
        self.action_space = spaces.Discrete(self.actions.forward + 1)

        self.reward_mode = 'default' # 'reward_model' 'none'

        # self.set_reward_model()

    # def set_reward_model(self):
        if reward_model is not None:
            self.reward_model = reward_model
        # else:
        #     self.reward_model = RewardModel()
        # self.goals_collected = 0
        # self.should_terminate = True

    @staticmethod
    def _gen_mission():
        return "Coin game"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # # Generate verical separation wall
        # for i in range(0, height):
        #     self.grid.set(5, i, Wall())
        
        # # Place the door and key
        # self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        # self.grid.set(3, 6, Key(COLOR_NAMES[0]))

        for i in range(self.n_walls):
            # Place a wall square in the bottom-right corner
            x = np.random.choice(range(1, width-1))
            y = np.random.choice(range(1, height-1))
            while self.grid.get(x, y) or (x, y) == self.agent_start_pos:
                x = np.random.choice(range(1, width-1))
                y = np.random.choice(range(1, height-1))
            # for j in range(0, 2):
            #     for k in range(0, 2):
            #         self.put_obj(Lava(), x+j, y+k)
            self.put_obj(Wall(), x, y)

        for i in range(0):
            # Place a lava square in the bottom-right corner
            x = np.random.choice(range(1, width-1))
            y = np.random.choice(range(1, height-1))
            while self.grid.get(x, y) or (x, y) == self.agent_start_pos:
                x = np.random.choice(range(1, width-1))
                y = np.random.choice(range(1, height-1))
            # for j in range(0, 2):
            #     for k in range(0, 2):
            #         self.put_obj(Lava(), x+j, y+k)
            self.put_obj(Lava(), x, y)

        if self.goal_pos is None:
            for i in range(1):
                # Place a goal square in the bottom-right corner
                x = np.random.choice(range(1, width-1))
                y = np.random.choice(range(1, height-1))
                # y = np.random.choice(range(1, height//2+1))
                while self.grid.get(x, y) or (x, y) == self.agent_start_pos:
                    x = np.random.choice(range(1, width-1))
                    y = np.random.choice(range(1, height-1))
                self.put_obj(Goal(), x, y)
        else:
            # self.put_obj(Goal(), width - 2, height - 2)
            self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])
        # self.target = np.array([x, y])

        # r = np.random.random()
        # if r < 1./3:
        # self.put_obj(Goal(), width - 2, height - 2)
        # elif r < 2./3:
        #     self.put_obj(Goal(), width // 2 - 1, height // 2 - 1)
        # else:
        #     self.put_obj(Goal(), 1, height // 2 - 1)

        # x = np.random.choice(range(1, width-1))
        # y = np.random.choice(range(1, height-1))
        # while self.grid.get(x, y):
        #     x = np.random.choice(range(1, width-1))
        #     y = np.random.choice(range(1, height-1))

        if self.agent_start_pos is None:
            x = np.random.choice(range(1, width-1))
            y = np.random.choice(range(1, height-1))
            while self.grid.get(x, y):
                x = np.random.choice(range(1, width-1))
                y = np.random.choice(range(1, height-1))
            self.agent_pos = (x, y)
        # Place the agent            
        else:
            self.agent_pos = self.agent_start_pos
        if self.agent_start_dir is not None:
            self.agent_dir = self.agent_start_dir
        else:
            self.agent_dir = np.random.choice(4)

        # self.mission = f"Coin game {self.target[0]} {self.target[1]}"
        self.mission = "Coin game"

    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """
        # reward = float(self.reward_model(self.obs) >= 0.5)
        # print(reward)
        if self.reward_mode == 'default':
            # print(1)
            return 2./5# - 0.9 * (self.step_count / self.max_steps)
        elif self.reward_mode == 'reward_model':
            state_action = self.reward_model.encode_action(self.obs, self.action)
            # reward = float(self.reward_model(self.obs) >= 0.5)
            reward = self.reward_model(state_action[None])[0]
            # reward = self.reward_model(self.obs)
            # print(reward)
            return reward
        else:
            # print(0)
            return 0 

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1
        self.action = action
        prev_dir = self.agent_dir
        self.prev_dir = prev_dir

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        self.fwd_cell = fwd_cell

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            reward = -0.9 * (1. / self.max_steps) # 1. / (self.width + self.height) # 
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                # self.grid.set(fwd_pos[0], fwd_pos[1], None)

                # # Place new goal
                # x = np.random.choice(range(1, self.width-1))
                # y = np.random.choice(range(1, self.height-1))
                # while self.grid.get(x, y):
                #     x = np.random.choice(range(1, self.width-1))
                #     y = np.random.choice(range(1, self.height-1))
                # self.put_obj(Goal(), x, y)

                # self.goals_collected += 1
                # if self.goals_remaining == 0:
                # terminated = self.should_terminate
                # terminated = True
                step_count = self.step_count
                self.reset()
                self.reset_counter += 1
                self.step_count = step_count
                # terminated = False
                # if fwd_pos[0] == self.target[0] \
                # and fwd_pos[1] == self.target[1]:
                # reward += self._reward() #/ self.num_goals
                # else:
                #     reward = -self._reward()
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                # terminated = True
                # self.reset()
                step_count = self.step_count
                self.reset()
                # self.reset_counter += 1
                self.step_count = step_count

        # # Pick up an object
        # elif action == self.actions.pickup:
        #     if fwd_cell and fwd_cell.can_pickup():
        #         if self.carrying is None:
        #             self.carrying = fwd_cell
        #             self.carrying.cur_pos = np.array([-1, -1])
        #             self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # # Drop an object
        # elif action == self.actions.drop:
        #     if not fwd_cell and self.carrying:
        #         self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
        #         self.carrying.cur_pos = fwd_pos
        #         self.carrying = None

        # # Toggle/activate an object
        # elif action == self.actions.toggle:
        #     if fwd_cell:
        #         fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        # if self.reward_mode == 'reward_model':
        #     reward = self._reward()

        if hasattr(self, 'reward_model') and self.reward_model.train:
            self.reward_model.observe(self.obs, action, reward) 

        obs = self.gen_obs()
        return obs, reward, terminated, truncated, {}

    # def reset(
    #     self,
    #     *,
    #     seed: int | None = None,
    #     options: dict[str, Any] | None = None,
    # ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
    #     obs, info = super().reset(seed=seed, options=options) # super().reset(seed=seed)
    #     return obs, info


class ComplexEnv(MiniGridEnv):
    def __init__(
        self,
        size=7,
        agent_view_size=13,
        agent_start_pos=None, # (1, 1),
        agent_start_dir=None, # 0,
        goal_pos=None, # (3, 3),
        max_steps: int | None = None,
        reward_model=None,
        n_walls=0,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = goal_pos
        self.n_walls = n_walls

        self.reset_counter = 0

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * (size-2)**2 // 5 # 20 timesteps

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # agent_view_size=agent_view_size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        # Allow only 3 actions permitted: left, right, forward
        self.action_space = spaces.Discrete(self.actions.forward + 1)

        self.reward_mode = 'default' # 'reward_model' 'none'

        # self.set_reward_model()

    # def set_reward_model(self):
        if reward_model is None:
            self.reward_model = RewardModel()
        else:
            self.reward_model = reward_model
        # self.goals_collected = 0
        # self.should_terminate = True

    @staticmethod
    def _gen_mission():
        return "Coin game"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # # Generate verical separation wall
        # for i in range(0, height):
        #     self.grid.set(5, i, Wall())
        
        # # Place the door and key
        # self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        # self.grid.set(3, 6, Key(COLOR_NAMES[0]))

        for i in range(self.n_walls):
            # Place a wall square in the bottom-right corner
            x = np.random.choice(range(1, width-1))
            y = np.random.choice(range(1, height-1))
            while self.grid.get(x, y) or (x, y) == self.agent_start_pos:
                x = np.random.choice(range(1, width-1))
                y = np.random.choice(range(1, height-1))
            # for j in range(0, 2):
            #     for k in range(0, 2):
            #         self.put_obj(Lava(), x+j, y+k)
            self.put_obj(Wall(), x, y)
 
        for i in range(0):
            # Place a lava square in the bottom-right corner
            x = np.random.choice(range(1, width-1))
            y = np.random.choice(range(1, height-1))
            while self.grid.get(x, y) or (x, y) == self.agent_start_pos:
                x = np.random.choice(range(1, width-1))
                y = np.random.choice(range(1, height-1))
            # for j in range(0, 2):
            #     for k in range(0, 2):
            #         self.put_obj(Lava(), x+j, y+k)
            self.put_obj(Lava(), x, y)

        if self.goal_pos is None:
            for i in range(1):
                # Place a goal square in the bottom-right corner
                x = np.random.choice(range(1, width-1))
                y = np.random.choice(range(1, height-1))
                # y = np.random.choice(range(height//2+1, height-1))
                while self.grid.get(x, y) or (x, y) == self.agent_start_pos:
                    x = np.random.choice(range(1, width-1))
                    y = np.random.choice(range(1, height-1))
                self.put_obj(Goal(), x, y)
        # self.target = np.array([x, y])
        else:
            # self.put_obj(Goal(), width - 2, height - 2)
            self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])

        if self.agent_start_pos is None:
            x = np.random.choice(range(1, width-1))
            y = np.random.choice(range(1, height-1))
            while self.grid.get(x, y):
                x = np.random.choice(range(1, width-1))
                y = np.random.choice(range(1, height-1))

        # # Place the agent
        # # if self.agent_start_pos is not None:
            self.agent_pos = (x, y)
        else:
            self.agent_pos = self.agent_start_pos
        if self.agent_start_dir is None:
            self.agent_dir = np.random.choice(4)
        else:
            self.agent_dir = self.agent_start_dir
            
        # else:
        #     self.place_agent()

        # self.mission = f"Coin game {self.target[0]} {self.target[1]}"
        self.mission = "Coin game"

    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """
        if self.reward_mode == 'default':
            # print(1)
            return 2./5# - 0.9 * (self.step_count / self.max_steps)
        elif self.reward_mode == 'reward_model':
            state_action = self.reward_model.encode_action(self.obs, self.action)
            # reward = float(self.reward_model(state_action) >= 0.5)
            reward = self.reward_model(state_action[None])[0]
            # reward = torch.round(reward)
            # reward = self.reward_model(state_action)
            # reward = self.reward_model(self.obs)
            # print(reward)
            return reward
        else:
            # print(0)
            return 0 

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1
        self.action = action
        prev_dir = self.agent_dir
        self.prev_dir = prev_dir

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        self.fwd_cell = fwd_cell

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            reward = -0.9 * (1. / self.max_steps)
            # reward = -0.9 * 1. / (self.width + self.height) # (1. / self.max_steps)
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                # self.grid.set(fwd_pos[0], fwd_pos[1], None)

                # # Place new goal
                # x = np.random.choice(range(1, self.width-1))
                # y = np.random.choice(range(1, self.height-1))
                # while self.grid.get(x, y):
                #     x = np.random.choice(range(1, self.width-1))
                #     y = np.random.choice(range(1, self.height-1))
                # self.put_obj(Goal(), x, y)

                # self.goals_collected += 1
                # if self.goals_remaining == 0:
                # terminated = self.should_terminate
                # terminated = True
                if self.reward_mode == 'reward_model':
                    # terminated = True
                    step_count = self.step_count
                    self.reset()
                    self.reset_counter += 1
                    self.step_count = step_count
                else:
                # if fwd_pos[0] == self.target[0] \
                # and fwd_pos[1] == self.target[1]:
                # reward += self._reward() #/ self.num_goals
                # else:
                #     reward = -self._reward()
                    reward = self._reward()
                    # terminated = True
                    step_count = self.step_count
                    self.reset()
                    self.reset_counter += 1
                    self.step_count = step_count
            if fwd_cell is not None and fwd_cell.type == "lava":
                # if self.reward_mode == 'reward_model':
                #     terminated = False
                # else:
                # terminated = True
                step_count = self.step_count
                self.reset()
                # self.reset_counter += 1
                self.step_count = step_count

        # # Pick up an object
        # elif action == self.actions.pickup:
        #     if fwd_cell and fwd_cell.can_pickup():
        #         if self.carrying is None:
        #             self.carrying = fwd_cell
        #             self.carrying.cur_pos = np.array([-1, -1])
        #             self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # # Drop an object
        # elif action == self.actions.drop:
        #     if not fwd_cell and self.carrying:
        #         self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
        #         self.carrying.cur_pos = fwd_pos
        #         self.carrying = None

        # # Toggle/activate an object
        # elif action == self.actions.toggle:
        #     if fwd_cell:
        #         fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        if self.reward_mode == 'reward_model':
            _reward = self._reward()
            # if _reward > 0.:
                # terminated = True
            reward += _reward
                # print(f'{reward.item():.2f}')

        obs = self.gen_obs()
        return obs, reward, terminated, truncated, {}

    # def reset(
    #     self,
    #     *,
    #     seed: int | None = None,
    #     options: dict[str, Any] | None = None,
    # ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
    #     obs, info = super().reset(seed=seed, options=options) # super().reset(seed=seed)
    #     return obs, info


# class NoRewardEnv(SimpleEnv):
#     def __init__(self, **kwargs):
#         super(NoRewardEnv, self).__init__(**kwargs)
#         # self.reward_model.train = False

#     def set_reward_model(self):
#         pass

#     def _reward(self):
#         return 0.

# class RewardModelEnv(SimpleEnv):
#     def __init__(self, reward_model, **kwargs):
#         super(RewardModelEnv, self).__init__(**kwargs)
#         self.reward_model = reward_model
#         self.reward_model.train = False

#     def set_reward_model(self):
#         pass

#     def _reward(self):
#         # super().observation()
#         # self.obs
#         return self.reward_model(self.obs)# self.action, self.actions, self.prev_dir, self.agent_dir, self.fwd_cell)




class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )
    #     self.num_objects = 4
    #     self.g = nn.Sequential(nn.Linear(features_dim // self.num_objects * 2, 128),
    #                             nn.ReLU())
    #     self.f = nn.Sequential(nn.Linear(128, 128),
    #                             nn.ReLU())

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    # def detect(self, observations: torch.Tensor) -> torch.Tensor:
    #     return self.linear(self.cnn(observations))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        ret = self.linear(self.cnn(observations))
        return ret



        # objs = torch.split(self.detect(observations), self.features_dim // self.num_objects, dim=-1)
        # # print(len(objs))
        # # Apply relational network
        # feat = 0
        # for i in range(self.num_objects):
        #     for j in range(self.num_objects):
        #         if i <= j:
        #             continue
        #         x = objs[i]
        #         y = objs[j]
        #         inp = torch.cat([x, y], dim=-1)
        #         feat += self.g(inp)
        # feat = self.f(feat)
        # return feat


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, num_steps: int, check_freq: int, log_dir: str, log_interval: int, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.num_steps = num_steps
        self.log_interval = log_interval

        eval_env = SimpleEnv(n_walls=3, render_mode="rgb_array")
        eval_env.reward_mode = 'default'
        # eval_env.reward_mode = 'default'
        # eval_env.reward_mode = 'reward_model'
        eval_env = FullyObsWrapper(eval_env)
        eval_env = Monitor(eval_env, self.log_dir)
        eval_env = DummyVecEnv([lambda: eval_env])
        eval_env = VecTransposeImage(eval_env)
        self.eval_env = eval_env

    # def _init_callback(self) -> None:
    #     # Create folder if needed
    #     if self.save_path is not None:
    #         os.makedirs(self.save_path, exist_ok=True)
    #     self.cnn = self.model.policy.features_extractor.cnn
    #     with torch.no_grad():
    #         n_flatten = self.cnn(torch.as_tensor(self.model.get_env().observation_space.sample()[None]).float()).shape[1]
    #     self.linear2 = nn.Sequential(nn.Linear(n_flatten, 1), nn.ReLU())

    #     self.loss_fn = nn.MSELoss()
    #     self.optimizer = optim.Adam(list(self.cnn.parameters()) + list(self.linear2.parameters()), lr=3e-4, eps=1e-5)
    #     self.logs = []

    def _on_step(self) -> bool:

        # if self.num_timesteps % 1e3:
        #     env = self.model.env.envs[0].env.env
        #     reward_model = env.reward_model
        #     if env.reward_mode == 'default':
        #         reward_model.train = False
        #         env.reward_mode = 'reward_model'
        #     else:
        #         reward_model.train = True
        #         env.reward_mode == 'default'

        # if self.num_timesteps == 5e3:
        #     # import pdb; pdb.set_trace()
        #     env = self.model.env.envs[0].env.env
        #     reward_model = env.reward_model
        #     logs = reward_model.logs
        #     plt.plot([s for s, l, x, y in logs], [l for s, l, x, y in logs])
        #     # plt.plot([s for s, x, y in self.model.env.unwrapped.logs], [y for s, x, y in self.model.env.unwrapped.logs])
        #     # for i, (s,x,y) in enumerate(self.model.env.logs):
        #     # i = 0
        #     # while i < len(callback.logs):
        #     #     j = i
        #     #     while j < len(callback.logs)-1 and callback.logs[j][0] < callback.logs[j+1][0]:
        #     #         j += 1
        #     #     print(callback.logs[i:j+1])
        #     #     plt.plot([s for s, x, y in callback.logs[i:j+1]], [y for s, x, y in callback.logs[i:j+1]])
        #     #     plt.plot([s for s, x, y in callback.logs[i:j+1]], [x for s, x, y in callback.logs[i:j+1]])
        #     #     i = j+1
        #     plt.title('Reward Model')
        #     plt.ylim(-0.009,1)
        #     plt.savefig('rewardmodel.png')

        #     y_true = [bool(y) for s, l, x, y in logs]
        #     y_pred = [x >= 0.5 for s, l, x, y in logs]
        #     mat = confusion_matrix(y_true, y_pred)
        #     disp = ConfusionMatrixDisplay(confusion_matrix=mat, display_labels=[False, True])
        #     print(mat)
        #     with open('reward_matrix.txt', 'w') as f:
        #         f.write(str(mat))
        #     disp.plot()
        #     plt.show()
        #     plt.savefig('reward_matrix.png')

        #     reward_model.train = False
        #     # env.reward_mode = 'default'
        #     env.reward_mode = 'reward_model'
        #     # env.reward_mode = 'none'
        #     # env.should_terminate = False
        #     # self.model._reward = lambda self: print(0); 0.
        #     # self._reward = lambda self: self.reward_model(self.obs)

        #     # self.model = PPO.load(self.save_path)
        #     # self.model.num_timesteps = self.num_timesteps + 1

        #     # rm_env = RewardModelEnv(reward_model, render_mode="rgb_array")
        #     # rm_env = FullyObsWrapper(rm_env)
        #     # rm_env = Monitor(rm_env, self.log_dir)
        #     # self.model.set_env(rm_env)
        #     # rm_env.reset()

        #     # no_env = NoRewardEnv(render_mode="rgb_array")
        #     # no_env = FullyObsWrapper(no_env)
        #     # no_env = Monitor(no_env, self.log_dir)
        #     # self.model.set_env(no_env)
        #     # no_env.reset()

        #     self.best_mean_reward = -np.inf
        #     return True

        # if self.num_timesteps > 1e4:
        #     import pdb; pdb.set_trace()

        # # import pdb; pdb.set_trace()
        # # observations = torch.as_tensor(self.locals['obs_tensor']).float()
        # observations = torch.as_tensor(self.locals['new_obs']).float()
        # pred = self.linear2(self.cnn(observations))

        # # grid = observations.clone()
        # # grid[observations[...,0] == OBJECT_TO_IDX["goal"],-1] = 8
        # goal_pos = np.where(observations[0,0] == OBJECT_TO_IDX["goal"])[:2]
        # agent_pos = np.where(observations[0,0] == OBJECT_TO_IDX["agent"])[:2]
        # agent_dir = observations[0,:,agent_pos[0], agent_pos[1]][-1]
        # # grid[observations[...,0] == OBJECT_TO_IDX["agent"],-1] = 10
        # # agent_dir = observations[observations[...,0] == OBJECT_TO_IDX["agent"],-1]
        # if self.n_calls > 5000:
        #     import pdb; pdb.set_trace()

        # agent_pos = np.asarray(agent_pos).squeeze()
        # goal_pos = np.asarray(goal_pos).squeeze()
        # # gt = (agent_pos + DIR_TO_VEC[int(agent_dir)] == goal_pos).all()
        # gt = (agent_pos == goal_pos).all()
        # gt = torch.Tensor([gt]).float().unsqueeze(0)
        # loss = self.loss_fn(pred, gt)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # self.logs.append([self.n_calls, pred.detach().squeeze().numpy(), gt.squeeze().numpy()])
        # print(loss, pred, gt, agent_pos, goal_pos)

        if self.log_interval is not None and self.n_calls % self.log_interval == 0:
            env = self.model.env.unwrapped.envs[0].unwrapped
            self.model.logger.record("rollout/ep_rew_mean_extrinsic", 0.4 * env.reset_counter / 20)
            env.reset_counter = 0

        #     # env = self.model.env

        #     # self.eval_env.reset()
        #     # model = PPO.load(callback.save_path)
        #     # self.model.env = self.eval_env
        #     # self.model.learn(1e4, reset_num_timesteps=False)
        #     mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=10)
        #     self.model.logger.record("rollout/ep_rew_mean_extrinsic", mean_reward)#[ep_info["r"] for ep_info in self.model.ep_info_buffer]))
        #     # self.model.env = env

        # if self.n_calls % self.check_freq == 0:

        #   # Retrieve training reward
        #   x, y = ts2xy(load_results(self.log_dir), 'timesteps')
        #   if len(x) > 0:
        #       # Mean training reward over the last 100 episodes
        #       mean_reward = np.mean(y[-100:])

        #       if self.verbose > 0:
        #         print("Num timesteps: {}".format(self.num_timesteps))
        #         print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

        #       # New best model, you could save the agent here
        #       if mean_reward > self.best_mean_reward:
        #           self.best_mean_reward = mean_reward
        #           # Example for saving best model
        #           if self.verbose > 0:
        #             print("Saving new best model to {}".format(self.save_path))
        #           self.model.save(self.save_path)

        return True

def custom_evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        observations, rewards, dones, infos = env.step(actions)
        # print(observations[...,0], actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            # import pdb; pdb.set_trace()
                            episode_rewards.append(info["episode"]["r"] > 0.)
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward



policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)


def main():

    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_dir = f"./ppo_minigrid_figure_1b-{args.mode}-{args.seed}/"
    os.makedirs(log_dir, exist_ok=True)

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            # config=vars(args),
            name=log_dir[6:-1], # f"minigrid_coins__{int(time.time())}",
            monitor_gym=True,
            save_code=True,
        )

    # env = SimpleEnv(render_mode="rgb_array")
    # # env = ComplexEnv(render_mode="rgb_array")
    # env = FullyObsWrapper(env)
    # env = Monitor(env, log_dir)

    # model = PPO("CnnPolicy", env, clip_range=.2, ent_coef=0.1, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=log_dir)
    # callback = SaveOnBestTrainingRewardCallback(1e3, check_freq=1e3, log_dir=log_dir)
    # model.learn(1e4, callback=callback, progress_bar=True, tb_log_name="single_run") # 2e5

    # model = PPO.load(callback.save_path)

    # eval_env = ComplexEnv(render_mode="rgb_array")
    # eval_env = FullyObsWrapper(eval_env)
    # eval_env = Monitor(eval_env, log_dir)

    # mean_reward, std_reward = custom_evaluate_policy(model, eval_env, n_eval_episodes=100)
    # print(mean_reward, std_reward)


    # if args.mode == 'reward-model':
    reward_model = RewardModel()
    env = SimpleEnv(n_walls=3, render_mode="rgb_array", reward_model=reward_model)
    # else:
        # env = SimpleEnv(n_walls=3, render_mode="rgb_array")
    # env = ComplexEnv(render_mode="rgb_array")
    env = FullyObsWrapper(env)
    env = Monitor(env, log_dir)

    # clip_range=.2, ent_coef=0.1, learning_rate=1e-4, ent_coef=1., 
    # model = PPO("CnnPolicy", env, n_steps=36, clip_range=.2, ent_coef=0.1, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=log_dir)
    model = PPO("CnnPolicy", env, learning_rate=1e-4, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=log_dir)
    callback = SaveOnBestTrainingRewardCallback(1e3, check_freq=1e3, log_dir=log_dir, log_interval=1e3)
    model.learn(12e4, callback=callback, progress_bar=True, tb_log_name="single_run") # 2e5

    # if args.mode == 'reward-model':
    reward_model.learn(n_steps=12e4) # //10)

    # env = SimpleEnv(n_walls=3, render_mode="rgb_array")
    # env = FullyObsWrapper(env)
    # env = Monitor(env, log_dir)
    # acc = reward_model.evaluate(env, model, n_eval_episodes=100)
    # print(f'Reward Model accuracy on simple environment: {acc.item():.2f}')

    # matrix, counts = reward_model.evaluate_enumerate(SimpleEnv, log_dir, acc.item(), 'accuracy_simple.png')

    if args.mode in ['reward-model', 'oracle']:
        eval_env = ComplexEnv(n_walls=3, render_mode="rgb_array", reward_model=reward_model)
    # elif args.mode in ['oracle', 'none']:
        # eval_env = ComplexEnv(n_walls=3, render_mode="rgb_array")
    elif args.mode in ['none', 'frozen']:
       eval_env = SimpleEnv(n_walls=3, render_mode="rgb_array", reward_model=reward_model)
    if args.mode == 'reward-model':
        eval_env.reward_mode = 'reward_model'
    elif args.mode == 'oracle':
        eval_env.reward_mode = 'default'
    elif args.mode == 'none':
        eval_env.reward_mode = 'none'
    eval_env = FullyObsWrapper(eval_env)
    eval_env = Monitor(eval_env, log_dir)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecTransposeImage(eval_env)
    eval_env.reset()

    # model = PPO.load(callback.save_path)
    model.env = eval_env
    model.learn(12e4, callback=callback, progress_bar=True, tb_log_name="single_run", reset_num_timesteps=False)

    # eval_env = ComplexEnv(n_walls=3, render_mode="rgb_array")
    # eval_env = FullyObsWrapper(eval_env)
    # eval_env = Monitor(eval_env, log_dir)
    # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
    # print(f'{mean_reward}\t{std_reward}')


if __name__ == "__main__":
    main()