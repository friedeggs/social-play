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

# seed = 1234
# seed = 1235
# seed = 1236
seed = 1237
# seed = 1238
np.random.seed(seed)
torch.manual_seed(seed)
np.set_printoptions(precision=2, suppress=True)

class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=7,
        agent_view_size=13,
        agent_start_pos=None,#(1, 1),
        agent_start_dir=None,#1,
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

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * (size-2)**2

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

        # self.put_obj(Wall(), 3, 2)
        # self.put_obj(Wall(), 3, 3)
        # self.put_obj(Wall(), 2, 4)

        # self.put_obj(Goal(), 4, 5)
        # self.put_obj(Goal(), 5, 3)

        # for i in range(0):
        #     # Place a lava square in the bottom-right corner
        #     x = np.random.choice(range(1, width-1))
        #     y = np.random.choice(range(1, height-1))
        #     while self.grid.get(x, y) or (x, y) == self.agent_start_pos:
        #         x = np.random.choice(range(1, width-1))
        #         y = np.random.choice(range(1, height-1))
        #     # for j in range(0, 2):
        #     #     for k in range(0, 2):
        #     #         self.put_obj(Lava(), x+j, y+k)
        #     self.put_obj(Lava(), x, y)

        if self.goal_pos is None:
            for i in range(1):
                # Place a goal square in the bottom-right corner
                x = np.random.choice(range(1, width-1))
                # y = np.random.choice(range(1, height-1))
                y = np.random.choice(range(1, height//2+1))
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


        # Place the agent
        if self.agent_start_pos is None:
            x = np.random.choice(range(1, width-1))
            y = np.random.choice(range(1, height-1))
            while self.grid.get(x, y):
                x = np.random.choice(range(1, width-1))
                y = np.random.choice(range(1, height-1))
            self.agent_pos = (x, y)
            self.agent_dir = np.random.choice(4)
        else:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        # import pdb; pdb.set_trace()
        # self.agent_dir = np.random.choice(4)
        # else:
        #     self.place_agent()

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
            return 1# - 0.9 * (self.step_count / self.max_steps)
        elif self.reward_mode == 'reward_model':
            reward = float(self.reward_model(self.obs) >= 0.5)
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

def main():
    env = SimpleEnv(n_walls=5, render_mode="human", highlight=False)
    env.reset()
    env.render()
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()