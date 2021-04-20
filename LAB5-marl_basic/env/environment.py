import numpy as np
import scipy.io as sio
import gym
from random import choice
from copy import deepcopy
import matplotlib.pyplot as plt


class Environment(object):
    """ each state is a 0-1 matrix, 
        where 0 denotes obstacle, 1 denotes space"""

    def __init__(self, args):
        self.args = args
        self.hist_len = args.hist_len  # 4
        self.image_dim = args.image_dim  # 32
        self.state_beta_dim = args.state_dim  # 3
        self.image_padding = args.image_padding
        self.max_train_doms = args.max_train_doms  # 6400
        self.start_valid_dom = args.start_valid_dom  # 6400
        self.start_test_dom = args.start_test_dom  # 7200
        self.step_reward = args.step_reward
        self.collision_reward = args.collision_reward
        self.terminal_reward = args.terminal_reward
        self.move = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])  # North, South, West, East
        self.last_train_dom = -1
        self.border_start = self.image_padding + 1  # >= 1 
        self.border_end = self.image_dim + self.image_padding - 2  # <= dim + pad - 2
        self.padded_state_shape = (self.image_dim + self.image_padding * 2, self.image_dim + self.image_padding * 2)
        self.state_alpha_dim = self.state_beta_dim + self.image_padding * 2
        self.pos_bias = np.array([self.image_padding, self.image_padding])
        self.load_data()

    def load_data(self):
        # ipdb.set_trace()
        data = sio.loadmat('data/gridworld_o_%d.mat' % self.image_dim)
        self.images = data['all_images']
        self.states_xy = data['all_states_xy_by_domain']
        self.max_domains = len(self.images)  # 8000
        self.preset_max_steps = {8: 38, 16: 86, 32: 178, 64: 246}
        if self.args.automax == 1:
            self.max_steps = self.image_dim * self.image_dim / 2
        elif self.args.automax == 2:
            self.max_steps = self.preset_max_steps[self.image_dim]
        else:
            self.max_steps = self.image_dim * self.args.max_steps

    def is_valid_pos(self, xy):
        # not in the border
        # return not (xy[0] >= self.image_dim-1 or xy[1] >= self.image_dim-1)
        return not (xy[0] > self.border_end or xy[0] < self.border_start or xy[1] > self.border_end or xy[
            1] < self.border_start)

    def restart(self, data_flag, init=False):
        # ipdb.set_trace()
        if data_flag == 'train':  # training
            init_dom = 0
            max_dom = self.max_train_doms
            if init:
                self.dom_ind = self.last_train_dom

        elif data_flag == 'valid':  # validation
            init_dom = self.start_valid_dom
            max_dom = self.start_test_dom
            if init:
                self.dom_ind = init_dom - 1

        else:  # testing
            init_dom = self.start_test_dom
            max_dom = self.max_domains
            if init:
                self.dom_ind = init_dom - 1

        invalid_flag = True
        pd = self.image_padding
        while invalid_flag:
            self.dom_ind += 1
            if self.dom_ind >= max_dom or self.dom_ind < init_dom:
                self.dom_ind = init_dom
            self.state = np.zeros(self.padded_state_shape, dtype=np.uint8)
            self.state[pd:-pd, pd:-pd] = self.images[self.dom_ind, 0]  # 32 * 32
            self.paths = self.states_xy[self.dom_ind, 0]
            for i in range(len(self.paths)):
                try:
                    self.a_xy = self.paths[i, 0][0] + self.pos_bias  # initial position of alpha
                    self.b_xy = self.paths[i, 0][-1] + self.pos_bias  # initial position of beta
                    self.min_steps = len(self.paths[i, 0]) / 2  # shortest path
                    # self.max_steps = len(self.paths[i, 0]) * self.args.max_steps
                except:
                    continue
                if self.is_valid_pos(self.a_xy) and self.is_valid_pos(self.b_xy):
                    invalid_flag = False
                    break

        self.terminal = False
        self.episode_reward = []
        self.states_alpha = np.zeros([1, self.hist_len, self.state_alpha_dim, self.state_alpha_dim], dtype=np.float32)
        self.states_beta = np.zeros([1, self.hist_len, self.state_beta_dim, self.state_beta_dim], dtype=np.float32)
        self.states_alpha[0, -1] = self.state[self.a_xy[0] - 1 - pd: self.a_xy[0] + 2 + pd,
                                   self.a_xy[1] - 1 - pd: self.a_xy[1] + 2 + pd]
        self.states_beta[0, -1] = self.state[self.b_xy[0] - 1: self.b_xy[0] + 2, self.b_xy[1] - 1: self.b_xy[1] + 2]

    def act(self, action, steps):
        act_a, act_b = divmod(action, 4)
        new_a_xy = self.a_xy + self.move[act_a]
        new_b_xy = self.b_xy + self.move[act_b]

        if self.is_valid_pos(new_a_xy) and self.state[new_a_xy[0], new_a_xy[1]] != 0:
            # not in the border and not obstacle
            self.a_xy = new_a_xy
            r_a = self.step_reward
        else:
            r_a = self.collision_reward

        if self.is_valid_pos(new_b_xy) and self.state[new_b_xy[0], new_b_xy[1]] != 0:
            self.b_xy = new_b_xy
            r_b = self.step_reward
        else:
            r_b = self.collision_reward

        # compute reward
        reward = r_a + r_b
        manhattan_distance = abs(sum(self.b_xy - self.a_xy))

        if self.args.use_instant_distance:
            r_ab = self.image_dim / (manhattan_distance + 1.0)
            reward += r_ab
        if manhattan_distance <= 1:
            reward += self.terminal_reward
        self.episode_reward.append([r_a, r_b, mahattan_distance])

        # terminal # distance = 0 or 1 means that alpha meets beta
        if manhattan_distance <= 1 or steps >= self.max_steps:
            self.terminal = True
        else:
            self.terminal = False

        # add current state to states history 
        pd = self.image_padding
        self.states_alpha[0, : -1] = self.states_alpha[0, 1:]
        self.states_alpha[0, -1] = self.state[self.a_xy[0] - 1 - pd: self.a_xy[0] + 2 + pd,
                                   self.a_xy[1] - 1 - pd: self.a_xy[1] + 2 + pd]

        self.states_beta[0, : -1] = self.states_beta[0, 1:]
        self.states_beta[0, -1] = self.state[self.b_xy[0] - 1: self.b_xy[0] + 2, self.b_xy[1] - 1: self.b_xy[1] + 2]

        return reward

    def getState(self):
        return self.states_alpha, self.states_beta

    def isTerminal(self):
        return self.terminal


class MultiTaskEnvironment(Environment):
    def __init__(self, args, role="alpha"):
        super(MultiTaskEnvironment, self).__init__(args)
        # self.goal_x = (self.border_end - self.border_start) / 2
        # self.goal_y = (self.border_end - self.border_start) / 2
        self.target_xy = np.array([8, 8]).reshape((2))
        self.map = np.zeros(self.padded_state_shape, dtype=np.uint8)
        self.pd = self.image_padding
        self.map[self.pd:-self.pd, self.pd:-self.pd] = self.images[args.map_num, 0]
        self.task_list = []
        for k in range(5):
            self.paths = self.states_xy[args.map_num + k, 0]
            # self.task_list = []
            for i in range(len(self.paths)):
                try:
                    a_xy = self.paths[i, 0][0] + self.pos_bias
                    if self.is_valid_pos(a_xy):
                        self.task_list.append(a_xy)
                    a_xy = self.paths[i, 0][-1] = self.pos_bias
                    if self.is_valid_pos(a_xy):
                        self.task_list.append(a_xy)
                except:
                    continue
        self.task_num = len(self.task_list)
        self.a_task_list = self.task_list[0: int(self.task_num / 2)]
        self.b_task_list = self.task_list[int(self.task_num / 2): self.task_num]
        self.role = role
        # print(self.role)
        self.terminal = False
        self.episode_reward = []
        self.states_agent = np.zeros([1, self.hist_len, self.state_alpha_dim, self.state_alpha_dim], dtype=np.float32)

    def restart(self, data_flag, init=False):
        self.terminal = False
        if self.role == "alpha":
            self.a_xy = choice(self.a_task_list)
        else:
            self.a_xy = choice(self.b_task_list)
        self.episode_reward = []
        self.states_agent[0, -1] = self.map[self.a_xy[0] - 1 - self.pd: self.a_xy[0] + 2 + self.pd,
                                   self.a_xy[1] - 1 - self.pd: self.a_xy[1] + 2 + self.pd]

    def act(self, action, steps):
        new_a_xy = self.a_xy + self.move[action]
        if self.is_valid_pos(new_a_xy) and self.map[new_a_xy[0], new_a_xy[1]] != 0:
            self.a_xy = new_a_xy
            reward = self.step_reward
        else:
            reward = self.collision_reward
        self.episode_reward.append(reward)
        manhatten_distance = abs(sum(self.target_xy - self.a_xy))
        distance = np.linalg.norm(self.target_xy - self.a_xy)
        if self.args.use_instant_distance:
            reward += self.image_dim / (manhatten_distance + 1.0)
        if distance <= 1:
            reward += self.terminal_reward
        self.episode_reward.append(reward)
        if distance <= 1 or steps >= self.max_steps:
            self.terminal = True
        else:
            self.terminal = False
        self.states_agent[0, : -1] = self.states_agent[0, 1:]
        self.states_agent[0, -1] = self.map[self.a_xy[0] - 1 - self.pd: self.a_xy[0] + 2 + self.pd,
                                   self.a_xy[1] - 1 - self.pd: self.a_xy[1] + 2 + self.pd]
        return reward

    def getState(self):
        return deepcopy(self.states_agent)

    def isTerminal(self):
        return deepcopy(self.terminal)


class OptimalRewardTestTask(object):
    def __init__(self):
        super(OptimalRewardTestTask, self).__init__()
        self.move = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
        self.map = np.zeros((4, 4))
        self.a_xy = np.array([0, 0], dtype=int)
        self.b_xy = np.array([3, 3], dtype=int)
        self.states_alpha = deepcopy(self.a_xy)
        self.states_beta = deepcopy(self.b_xy)
        self.joint_state = np.concatenate([self.states_alpha, self.states_beta])
        self.terminal = False
        self.step_num = 0
        self.max_step_num = 80

    def _is_valid_pos(self, xy):
        # not in the border
        # return not (xy[0] >= self.image_dim-1 or xy[1] >= self.image_dim-1)
        return not (xy[0] > 3 or xy[0] < 0 or xy[1] > 3 or xy[1] < 0)

    def restart(self):
        self.map = np.zeros((4, 4))
        self.a_xy = np.array([0, 0], dtype=int)
        self.b_xy = np.array([3, 3], dtype=int)
        self.states_alpha = deepcopy(self.a_xy)
        self.states_beta = deepcopy(self.b_xy)
        self.joint_state = np.concatenate([self.states_alpha, self.states_beta])
        self.terminal = False
        self.step_num = 0

    def act(self, action):
        act_a, act_b = divmod(action, 4)
        new_a_xy = self.a_xy + self.move[act_a]
        new_b_xy = self.b_xy + self.move[act_b]
        if self._is_valid_pos(new_a_xy):
            self.a_xy = new_a_xy
        if self._is_valid_pos(new_b_xy):
            self.b_xy = new_b_xy
        r_a = -1
        r_b = -1
        # reward = (r_a + r_b) / 2
        manhattan_distance = sum(abs(self.b_xy - self.a_xy))
        reward = -manhattan_distance
        if manhattan_distance <= 1:
            self.terminal = True
        else:
            self.terminal = False
        self.step_num += 1
        if self.step_num >= self.max_step_num:
            self.terminal = True
        self.states_alpha = deepcopy(self.a_xy)
        self.states_beta = deepcopy(self.b_xy)
        self.joint_state = np.concatenate([self.states_alpha, self.states_beta])
        return reward

    def getState(self):
        return self.states_alpha, self.states_beta, self.joint_state

    def isTerminal(self):
        return self.terminal


map = np.zeros((4, 4))
plt.pcolormesh(map, cmap=plt.cm.gray, vmin=-2, vmax=1, edgecolors="k", linewidth=2)
point_list = []
plt.scatter(x=0, y=0, s=180, marker=".", c="red", linewidths=15)
plt.scatter(x=3, y=3, s=180, marker=".", c="blue", linewidths=15)

plt.xlim((-1, 5))
plt.ylim((-1, 5))
plt.savefig("./map.pdf", format="pdf", dpi=1000)