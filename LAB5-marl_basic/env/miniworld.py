from env import Environment
from env import process_gridworld_data
import argparse
from env import MultiTaskEnvironment, OptimalRewardTestTask
import gym
import random
import numpy as np
import matplotlib.pyplot as plt


class Miniworld(gym.Env):
    def __init__(self, role="alpha"):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--image_dim", type=int, default=16, help="")
        self.parser.add_argument("--state_dim", type=int, default=3, help="")
        self.parser.add_argument("--hist_len", type=int, default=2, help="")
        # self.parser.add_argument("--max_steps", type=int, default=87, help="")
        self.parser.add_argument("--image_padding", type=int, default=1, help="")
        self.parser.add_argument("--max_train_doms", type=int, default=6400, help="")
        self.parser.add_argument("--start_valid_dom", type=int, default=6400, help="")
        self.parser.add_argument("--start_test_dom", type=int, default=7200, help="")
        self.parser.add_argument("--automax", type=int, default=2, help="")
        self.parser.add_argument("--autolen", type=int, default=1, help="")
        self.parser.add_argument("--use_instant_distance", type=int, default=0, help="")
        self.parser.add_argument("--step_reward", type=float, default=-1.0, help="")
        self.parser.add_argument("--collision_reward", type=float, default=-10.0, help="")
        self.parser.add_argument("--terminal_reward", type=float, default=+500.0, help="")
        self.parser.add_argument("--map_num", type=int, default=0, help="")
        self.args = self.parser.parse_args()
        self.inner_env = MultiTaskEnvironment(self.args, role=role)
        self.step_num = 0
        self.inner_env.restart(data_flag='train', init=True)
        self.action_space = gym.spaces.Discrete(4)
        wow = self.inner_env.getState().reshape((-1)).shape
        high = np.ones(shape=wow, dtype=int)
        low = np.zeros(shape=wow, dtype=int)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.int)

    def _get_reward(self, done):
        if done:
            reward = 1.0 - (float(self.step_num) / float(self.get_max_steps()))
        else:
            reward = 0.0
        return reward

    def step(self, action):
        self.step_num += 1
        reward = self.inner_env.act(action, self.step_num)
        obs = self.inner_env.getState().reshape((-1))
        done = self.inner_env.isTerminal()
        #if self.step_num >= self.get_max_steps():
        #    done = True
        reward = self._get_reward(done)
        return obs, reward, done, {}

    def reset(self):
        self.step_num = 0
        self.inner_env.restart(data_flag='train', init=True)
        obs = self.inner_env.getState().reshape((-1))
        return obs

    def get_agent_loc(self):
        return self.inner_env.a_xy

    def _get_state(self):
        return self.inner_env.getState().reshape((-1))

    def get_max_steps(self):
        return self.inner_env.max_steps


class TestEnv(gym.Env):
    def __init__(self, joint=True, share_obs=False):
        self.inner_env = OptimalRewardTestTask()
        self.step_num = 0
        self.terminal = False
        self.is_joint_action = joint
        self.share_obs = share_obs


    def step(self, action):
        self.step_num += 1
        reward = self.inner_env.act(action)
        obs_a, obs_b, obs_j = self.inner_env.getState()
        if self.is_joint_action:
            obs = obs_j
        else:
            obs = [obs_a, obs_b]
        if self.share_obs:
            obs = [obs_j, obs_j]
        done = self.inner_env.isTerminal()
        return obs, reward, done, {}

    def reset(self):
        self.step_num = 0
        self.inner_env.restart()
        self.terminal = False
        obs_a, obs_b, obs_j = self.inner_env.getState()
        if self.is_joint_action:
            obs = obs_j
        else:
            obs = [obs_a, obs_b]
        if self.share_obs:
            obs = [obs_j, obs_j]
        return obs

    def _get_state(self):
        return self.inner_env.getState().reshape((-1))

    def get_max_steps(self):
        return self.inner_env.max_step_num
