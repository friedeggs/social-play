from __future__ import annotations

import glob
import os
import time

import gymnasium as gym

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy

from gymnasium.envs.registration import register
register(
    id='multigrid-prosocial-v0',
    entry_point='gym-multigrid.gym_multigrid.envs:ProsocialEnv',
)

import random

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


from gymnasium import spaces
from gymnasium.core import ObservationWrapper
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX

from gym_multigrid.multigrid import World


from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

# from minigrid.wrappers import ImgObsWrapper, NoDeath, RGBImgPartialObsWrapper

from typing import Any, Iterable, SupportsFloat, TypeVar
from typing import Callable, Dict, List, Optional, Tuple, Union

from gymnasium.core import ActType, ObsType, WrapperObsType

import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback

from gymnasium.core import ActionWrapper, Wrapper
from minigrid.core.constants import STATE_TO_IDX


import warnings


from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped#, VecTransposeImage
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvWrapper


import wandb

# seed = 1234
# seed = 1235
seed = 1236
np.random.seed(seed)
torch.manual_seed(seed)
np.set_printoptions(precision=2, suppress=True)




# class VecTransposeImage(VecEnvWrapper):
#     """
#     Re-order channels, from HxWxC to CxHxW.
#     It is required for PyTorch convolution layers.

#     :param venv:
#     :param skip: Skip this wrapper if needed as we rely on heuristic to apply it or not,
#         which may result in unwanted behavior, see GH issue #671.
#     """

#     def __init__(self, venv: VecEnv, skip: bool = False):
#         assert is_image_space(venv.observation_space) or isinstance(
#             venv.observation_space, spaces.dict.Dict
#         ), "The observation space must be an image or dictionary observation space"

#         self.skip = skip
#         # Do nothing
#         if skip:
#             super().__init__(venv)
#             return

#         if isinstance(venv.observation_space, spaces.dict.Dict):
#             self.image_space_keys = []
#             observation_space = deepcopy(venv.observation_space)
#             for key, space in observation_space.spaces.items():
#                 if is_image_space(space):
#                     # Keep track of which keys should be transposed later
#                     self.image_space_keys.append(key)
#                     observation_space.spaces[key] = self.transpose_space(space, key)
#         else:
#             observation_space = self.transpose_space(venv.observation_space)
#         super().__init__(venv, observation_space=observation_space)

#     @staticmethod
#     def transpose_space(observation_space: spaces.Box, key: str = "") -> spaces.Box:
#         """
#         Transpose an observation space (re-order channels).

#         :param observation_space:
#         :param key: In case of dictionary space, the key of the observation space.
#         :return:
#         """
#         # Sanity checks
#         assert is_image_space(observation_space), "The observation space must be an image"
#         assert not is_image_space_channels_first(
#             observation_space
#         ), f"The observation space {key} must follow the channel last convention"
#         height, width, channels = observation_space.shape
#         new_shape = (channels, height, width)
#         return spaces.Box(low=0, high=255, shape=new_shape, dtype=observation_space.dtype)

#     @staticmethod
#     def transpose_image(image: np.ndarray) -> np.ndarray:
#         """
#         Transpose an image or batch of images (re-order channels).

#         :param image:
#         :return:
#         """
#         if len(image.shape) == 3:
#             return np.transpose(image, (2, 0, 1))
#         return np.transpose(image, (0, 3, 1, 2))

#     def transpose_observations(self, observations: Union[np.ndarray, Dict]) -> Union[np.ndarray, Dict]:
#         """
#         Transpose (if needed) and return new observations.

#         :param observations:
#         :return: Transposed observations
#         """
#         # Do nothing
#         if self.skip:
#             return observations

#         if isinstance(observations, dict):
#             # Avoid modifying the original object in place
#             observations = deepcopy(observations)
#             for k in self.image_space_keys:
#                 observations[k] = self.transpose_image(observations[k])
#         else:
#             observations = self.transpose_image(observations)
#         return observations

#     def step_wait(self) -> VecEnvStepReturn:
#         observations, rewards, dones, infos = self.venv.step_wait()

#         # Transpose the terminal observations
#         for idx, done in enumerate(dones):
#             if not done:
#                 continue
#             if "terminal_observation" in infos[idx]:
#                 infos[idx]["terminal_observation"] = self.transpose_observations(infos[idx]["terminal_observation"])

#         return self.transpose_observations(observations), rewards, dones, infos

#     def reset(self) -> Union[np.ndarray, Dict]:
#         """
#         Reset all environments
#         """
#         return self.transpose_observations(self.venv.reset())

#     def close(self) -> None:
#         self.venv.close()


class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        ## Reward model
        observation_space = spaces.Box(
            low=0,
            high=255,
            # shape=(13, 13, 5),  # number of cells
            shape=(9, 9, 9),  # number of cells
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

    def encode_actions(self, state, actions):
        action_enc = np.expand_dims(state[0,...,0] == OBJECT_TO_IDX["agent"], -1)
        action_enc = action_enc.astype(np.float32)
        action_enc0 = action_enc * actions[0]
        action_enc1 = action_enc * actions[1]
        state_action = np.concatenate([state[0], action_enc0, action_enc1], axis=-1)
        # import pdb; pdb.set_trace()
        return state_action

    def observe(self, state, actions, reward):
        if reward > 0.:
            self.num_reward += 1
            # state_action = self.encode_action(state, actions[0])
            # self.data.append((state_action, actions, reward))
            # state_action = self.encode_action(state, 0)
            # self.data.append((state_action, 0, 0))
            # state_action = self.encode_action(state, 1)
            # self.data.append((state_action, 1, 0))
        # else:
        state_actions = self.encode_actions(state, actions)
        # state_actions = self.encode_action(state_action, actions[1])
        self.data.append((state_actions, actions, reward))

    def learn(self, n_steps=1e4):
        # num_reward = self.num_reward
        self.train_data = self.data#[:len(self.data)*9//10]
        # self.test_data = self.data[len(self.data)*9//10:]
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

        # num_reward = sum([1 if reward > 0. else 0 for _, _, reward in self.test_data])
        # num_data = len(self.test_data)
        # frac_rew = 1./(2*num_reward)
        # frac = 1./(2*(num_data-num_reward))
        # test_weights = [frac_rew if reward > 0. else frac for _, _, reward in self.test_data]

        num_epochs = 10
        batch_size = 10
        # writer = torch.utils.tensorboard.SummaryWriter(log_dir="log_reward_model")
        train_sampler = torch.utils.data.WeightedRandomSampler(train_weights, num_samples=len(self.train_data), replacement=True)
        train_dataloader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, sampler=train_sampler, drop_last=True)
        # test_sampler = torch.utils.data.WeightedRandomSampler(test_weights, num_samples=len(self.test_data), replacement=True)
        # test_dataloader = torch.utils.data.DataLoader(self.test_data, batch_size=batch_size, sampler=test_sampler, drop_last=True)
        # for i in tqdm(range(int(n_steps))):
        for epoch_id in range(num_epochs):
            for i, batch in tqdm(enumerate(train_dataloader)):
                state_action, action, reward = batch
                pred, loss = self.forward(state_action, reward)

                # if i >= n_steps:
                #     break



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

class VecTransposeImage(VecEnvWrapper):
    def __init__(self, venv: VecEnv, skip: bool = False):
        # assert is_image_space(venv.observation_space) or isinstance(
        #     venv.observation_space, spaces.dict.Dict
        # ), "The observation space must be an image or dictionary observation space"

        self.skip = skip
        # Do nothing
        if skip:
            super().__init__(venv)
            return

        # if isinstance(venv.observation_space, spaces.dict.Dict):
        #     self.image_space_keys = []
        #     observation_space = deepcopy(venv.observation_space)
        #     for key, space in observation_space.spaces.items():
        #         if is_image_space(space):
        #             # Keep track of which keys should be transposed later
        #             self.image_space_keys.append(key)
        #             observation_space.spaces[key] = self.transpose_space(space, key)
        # else:
        observation_space = self.transpose_space(venv.observation_space)
        super().__init__(venv, observation_space=observation_space)


    @staticmethod
    def transpose_space(observation_space: spaces.Box, key: str = "") -> spaces.Box:
        """
        Transpose an observation space (re-order channels).

        :param observation_space:
        :param key: In case of dictionary space, the key of the observation space.
        :return:
        """
        # Sanity checks
        # assert is_image_space(observation_space), "The observation space must be an image"
        # assert not is_image_space_channels_first(
        #     observation_space
        # ), f"The observation space {key} must follow the channel last convention"
        num_agents, height, width, channels = observation_space.shape
        new_shape = (num_agents, channels, height, width)
        return spaces.Box(low=0, high=255, shape=new_shape, dtype=observation_space.dtype)

    def transpose_observations(self, obs):
        # return obs.transpose(0,1,4,2,3)#[:,0]
        return obs


    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()

        # Transpose the terminal observations
        for idx, done in enumerate(dones):
            if not done:
                continue
            if "terminal_observation" in infos[idx]:
                infos[idx]["terminal_observation"] = self.transpose_observations(infos[idx]["terminal_observation"])

        return self.transpose_observations(observations), rewards, dones, infos


    def reset(self) -> Union[np.ndarray, Dict]:
        """
        Reset all environments
        """
        # import pdb; pdb.set_trace()
        return self.transpose_observations(self.venv.reset())


# class FullyObsTransposeWrapper(ObservationWrapper):
#     """
#     Fully observable gridworld using a compact grid encoding instead of the agent view.

#     Example:
#         >>> import gymnasium as gym
#         >>> import matplotlib.pyplot as plt
#         >>> from minigrid.wrappers import FullyObsWrapper
#         >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
#         >>> obs, _ = env.reset()
#         >>> obs['image'].shape
#         (7, 7, 3)
#         >>> env_obs = FullyObsWrapper(env)
#         >>> obs, _ = env_obs.reset()
#         >>> obs['image'].shape
#         (11, 11, 3)
#     """

#     def __init__(self, env):
#         super().__init__(env)

#         new_image_space = spaces.Box(
#             low=0,
#             high=255,
#             shape=(2, self.env.width*2-5, self.env.height*2-5, 7),  # number of cells
#             dtype="uint8",
#         )

#         self.observation_space = new_image_space

#     def observation(self, obs):
#         env = self.unwrapped

#         partial_grids = []
#         for i in range(1): # len(env.agents)):
#             full_grid = env.grid.encode(env.world)
#             # full_grid[env.agents[i].pos[0]][env.agents[i].pos[1]] = np.array(
#             #     [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agents[i].dir]
#             # )
#             w = self.env.unwrapped.width-1-1
#             h = self.env.unwrapped.height-1-1

#             full_grid = np.pad(full_grid, ((w, h), (w, h), (0, 0)), 'constant', constant_values=0)
#             partial_grid = full_grid[env.agents[i].pos[0]+1+w-w:env.agents[i].pos[0]+w+w,env.agents[i].pos[1]+1+h-h:env.agents[i].pos[1]+h+h]
#             for i in range(env.agents[i].dir):
#                 partial_grid = np.rot90(partial_grid, k=1, axes=(1,0))
#             partial_grid = np.rot90(partial_grid, k=1, axes=(1,0))
#             partial_grid = np.rot90(partial_grid, k=1, axes=(1,0))
#             partial_grid = np.flip(partial_grid, axis=1).copy()

#             is_goal = np.expand_dims(partial_grid[...,0] == OBJECT_TO_IDX["goal"], -1)
#             partial_grid = np.concatenate([partial_grid, is_goal], axis=-1)
#             partial_grids.append(partial_grid)

#         for i in range(1, len(env.agents)):
#             full_grid = env.grid.encode(env.world)
#             full_grid = np.pad(full_grid, ((1, 1), (1, 1), (0, 1)), 'constant', constant_values=0)
#             partial_grid = full_grid
#             partial_grids.append(partial_grid)

#             # import pdb; pdb.set_trace()

#         partial_grids = np.array(partial_grids)
#         obs = partial_grids
#         return obs.permute(0,1,4,2,3)[:,0]
#         # return partial_grids

#     def step(
#         self, actions: List[ActType]
#     ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
#         """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.observation` on the returned observations."""
#         obs = self.env.gen_obs()
#         self.env.obs = self.observation(obs)

#         observation, rewards, terminated, truncated, info = self.env.step(actions)

#         # import pdb; pdb.set_trace()
#         if self.reward_model is not None and not self.reward_model.train:
#             state_actions = self.encode_actions(self.obs, actions)
#             reward = self.reward_model(state_actions)
#         else:
#             reward = rewards[0]

#         if self.reward_model is not None and self.reward_model.train:
#             self.reward_model.observe(self.obs, actions, reward)

#         obs = self.observation(observation)
#         return obs, reward, terminated, truncated, info

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
            shape=(2, self.env.width*2-5, self.env.height*2-5, 7),  # number of cells
            dtype="uint8",
        )

        self.observation_space = new_image_space

    def observation(self, obs):
        env = self.unwrapped

        partial_grids = []
        for i in range(1): # len(env.agents)):
            full_grid = env.grid.encode(env.world)
            # full_grid[env.agents[i].pos[0]][env.agents[i].pos[1]] = np.array(
            #     [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agents[i].dir]
            # )
            w = self.env.unwrapped.width-1-1
            h = self.env.unwrapped.height-1-1

            full_grid = np.pad(full_grid, ((w, h), (w, h), (0, 0)), 'constant', constant_values=0)
            partial_grid = full_grid[env.agents[i].pos[0]+1+w-w:env.agents[i].pos[0]+w+w,env.agents[i].pos[1]+1+h-h:env.agents[i].pos[1]+h+h]
            for i in range(env.agents[i].dir):
                partial_grid = np.rot90(partial_grid, k=1, axes=(1,0))
            partial_grid = np.rot90(partial_grid, k=1, axes=(1,0))
            partial_grid = np.rot90(partial_grid, k=1, axes=(1,0))
            partial_grid = np.flip(partial_grid, axis=1).copy()

            is_goal = np.expand_dims(partial_grid[...,0] == OBJECT_TO_IDX["goal"], -1)
            partial_grid = np.concatenate([partial_grid, is_goal], axis=-1)
            partial_grids.append(partial_grid)

        for i in range(1, len(env.agents)):
            full_grid = env.grid.encode(env.world)
            full_grid = np.pad(full_grid, ((1, 1), (1, 1), (0, 1)), 'constant', constant_values=0)
            partial_grid = full_grid
            partial_grids.append(partial_grid)

            # import pdb; pdb.set_trace()

        partial_grids = np.array(partial_grids)
        return partial_grids

    def step(
        self, actions: List[ActType]
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.observation` on the returned observations."""
        obs = self.env.gen_obs()
        self.env.obs = self.observation(obs)

        observation, rewards, terminated, truncated, info = self.env.step(actions)

        
        if self.reward_model is not None and not self.reward_model.train:
            state_actions = self.reward_model.encode_actions(self.obs, actions)
            # import pdb; pdb.set_trace()
            reward = self.reward_model(state_actions[None])[0]
        else:
            reward = rewards[0]

        if self.reward_model is not None and self.reward_model.train:
            self.reward_model.observe(self.obs, actions, reward)

        obs = self.observation(observation)
        return obs, reward, terminated, truncated, info



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

    def _on_step(self) -> bool:

        if self.log_interval is not None and self.n_calls % self.log_interval == 0:
            env = self.model.env.unwrapped.envs[0].unwrapped
            # import pdb; pdb.set_trace()
            self.model.logger.record("rollout/ep_rew_mean_extrinsic", 2.7 * env.reset_counter / sum([1 for ep_info in self.model.ep_info_buffer]))
            env.reset_counter = 0

        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])

              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[-1]
        # print(observation_space.shape)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[0]).float().permute(2,0,1)[None]).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # print(observations.shape) # ([1, 7, 9, 9])
        # import pdb; pdb.set_trace()
        ret = self.linear(self.cnn(observations))
        return ret


class DFSAgent:
    def __init__(self, index):
        self.id = index

    def get_action(self, obs, step=None):
        return other_fn(obs)

def other_fn(obs):
    # action = np.random.choice(6)

    # if obs.dim() == 4:
    #     obs = obs[0]

    if obs.shape[0] == 7:
        obs = obs.permute(1,2,0)

    obs = torch.tensor(obs)

    positions = np.where(obs[...,0] == OBJECT_TO_IDX["agent"])
    # print(positions)
    agent0 = obs[positions[0][0],positions[1][0]]
    # agent1 = obs[positions[0][1],positions[1][1]]
    if agent0[1] == 1:
        pos = np.array([positions[0][0],positions[1][0]])
    else:
        pos = np.array([positions[0][1],positions[1][1]])
    # pos = np.where(obs[:,:] == agent1)
    pos = tuple(pos)
    # print(pos)
    goal_pos = np.array(np.where(obs[...,0] == OBJECT_TO_IDX["goal"])).reshape(-1)

    # goal_pos = [4, 4]
    # print(goal_pos)

    def dfs(cur_loc, path):
        if cur_loc == tuple(goal_pos):
            return path + [cur_loc]
        for neighbor in [(cur_loc[0],cur_loc[1]-1),(cur_loc[0],cur_loc[1]+1),(cur_loc[0]-1,cur_loc[1]),(cur_loc[0]+1,cur_loc[1])]:
            if neighbor in path or obs[neighbor[0],neighbor[1],0] == 2: continue
            res = dfs(neighbor, path + [cur_loc])
            if res:
                return res
        return []

    # import pdb; pdb.set_trace()
    path = dfs(pos, [])
    # print(path)
    actions = []

    caregiver_start_dir = obs[pos[0],pos[1],-3]#.to(dtype=torch.int32)
    VEC_TO_DIR = {(x[0], x[1]): i for i, x in enumerate(DIR_TO_VEC)}

    # turns
    prev_loc = path[0]
    prev_dir = caregiver_start_dir
    for cur_loc in path[1:]:
        vec = (cur_loc[0]-prev_loc[0], cur_loc[1]-prev_loc[1])
        cur_dir = VEC_TO_DIR[vec]
        if cur_dir != prev_dir:
            # add turns 
            while cur_dir > prev_dir:
                actions.append(2) # TODO
                prev_dir += 1
            while cur_dir < prev_dir:
                actions.append(1) # TODO
                prev_dir -= 1
            actions.append(3)
        else:
            actions.append(3)
        prev_dir = cur_dir
        prev_loc = cur_loc

    # print(actions)

    action = actions[0]
    # action = 5
    # print('dir: ', caregiver_start_dir)

    # import pdb; pdb.set_trace()
    if obs.dim() == 5:
        batch_size = obs.shape[0]
        return torch.ones((batch_size, 1)) * action
    return torch.Tensor([action]).squeeze()

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

def main():
    # log_dir = "./ppo_minigrid_figure_8"
    log_dir = f"./ppo_minigrid_figure_8-prosocial-24e4-reward_model-epochs-10-{seed}/"
    save_path = os.path.join(log_dir, f"figure_8-prosocial-24e4-reward_model-epochs-10-{seed}")

    os.makedirs(log_dir, exist_ok=True)

    # env = SimpleEnv(n_walls=3, render_mode="human", highlight=False)
    # reward_model = RewardModel()
    env = gym.make("multigrid-prosocial-v0")#, render_mode="human")
    env = FullyObsWrapper(env)
    env = Monitor(env, log_dir)

    next_obs, _ = env.reset()
    next_obs, _ = env.reset()
    next_obs, _ = env.reset()
    next_obs, _ = env.reset()
    next_obs, _ = env.reset()
    next_obs, _ = env.reset()
    next_obs, _ = env.reset()
    next_obs, _ = env.reset()#
    next_obs, _ = env.reset()
    next_obs, _ = env.reset()
    next_obs, _ = env.reset()
    env.reset_counter = 0
    # print(next_obs)
    # next_obs = torch.Tensor(next_obs)
    env.render()
    import pdb; pdb.set_trace()

    # # num_agents = 2
    # # agents = [DFSAgent(_) for _ in range(num_agents)]
    # # ent_coef=0.1,
    # model = PPO("CnnPolicy", env, learning_rate=1e-4, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=log_dir)
    # callback = SaveOnBestTrainingRewardCallback(1e3, check_freq=1e3, log_dir=log_dir, log_interval=1e3)
    # model.learn(48e4, callback=callback, progress_bar=True, tb_log_name="single_run", other_fn=other_fn) # 2e5

    # model.save(callback.save_path)

    # reward_model.learn()

    # model = PPO.load(callback.save_path)

    # reward_model.train = False
    # env = gym.make("multigrid-prosocial-v0", reward_model=reward_model)
    # env = FullyObsWrapper(env)
    # env = Monitor(env, log_dir)
    # env = DummyVecEnv([lambda: env])
    # env = VecTransposeImage(env)
    # env.reset()
    # env.reset_counter = 0
    # # next_obs, _ = env.reset(seed=1234)

    # model.env = env
    # callback = SaveOnBestTrainingRewardCallback(1e3, check_freq=1e3, log_dir=log_dir, log_interval=1e3)
    # model.learn(48e4, callback=callback, progress_bar=True, tb_log_name="single_run", other_fn=other_fn, reset_num_timesteps=False)

    # model.save(save_path)

    # model = PPO.load(save_path)

    # env = gym.make("multigrid-prosocial-v0", render_mode="human", reward_model=reward_model)
    # env = FullyObsWrapper(env)
    # env = Monitor(env, log_dir)
    # next_obs, _ = env.reset(seed=1234)
    # # next_obs = torch.Tensor(next_obs)

    # for step in range(100):
    #     actions = [None for i in range(num_agents)]
    #     actions[0], _ = model.predict(next_obs[0], other_fn=other_fn)
    #     # # import pdb; pdb.set_trace()
    #     actions[0] = torch.Tensor([actions[0].item()])
    #     # actions[0] = torch.Tensor([5])
    #     for i in range(1, num_agents):
    #         action = agents[i].get_action(next_obs[i])
    #         actions[i] = torch.unsqueeze(action, 0)
    #     # print(actions[1])
    #     # actions[1] = torch.Tensor([5])
    #     actions = torch.cat(actions, dim=0)

    #     next_obs, rewards, next_done, next_truncated, infos = env.step(actions.cpu().numpy())
    #     # next_obs = torch.Tensor(np.array(next_obs))

    #     # print(next_obs.shape) # 2, 7, 7, 6
    #     # [2, 9, 9, 7]
    #     # import pdb; pdb.set_trace()

    #     env.render()

    #     if next_done or next_truncated:
    #         next_obs, _ = env.reset()




if __name__ == "__main__":
    main()
