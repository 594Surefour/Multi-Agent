import numpy as np
import copy
import ipdb
import random

class LearningAgent():
    def __init__(self, 
                 aid=None, 
                 alpha=0.2, 
                 policy=None, 
                 gamma=0.99, 
                 actions=None, 
                 alpha_decay_rate=None, 
                 epsilon_decay_rate=None):

        self.aid = aid
        self.alpha = alpha
        self.gamma = gamma
        self.policy = policy
        self.reward_history = []
        self.name = "agent"
        self.actions = actions
        self.gamma = gamma
        self.alpha_decay_rate = alpha_decay_rate
        self.epsilon_decay_rate = epsilon_decay_rate
        self.previous_action_id = None
        self.q_values = self._init_q_values()

    def _init_q_values(self):
        q_values = np.repeat(0.0, len(self.actions))
        return q_values

    def init_policy(self, policy):
        self.policy = policy

    def act(self, training=True):
        if training:
            action_id = self.policy.select_action(self.q_values)
            self.previous_action_id = action_id
            action = self.actions[action_id]
        else:
            action_id = self.policy.select_greedy_action(self.q_values)
            action = self.actions[action_id]
        return action

    def get_previous_action(self):
        action = self.actions[self.previous_action_id]
        return action

    def observe(self, reward, is_learn=True):
        if is_learn:
            self.learn(reward)

    def learn(self, reward, is_finish=True):
        self.reward_history.append(reward)
        self.q_values[self.previous_action_id] = self.compute_q_value(reward)

    def compute_q_value(self, reward):
        q = self.q_values[self.previous_action_id] # Q(s, a)
        # Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        updated_q = q + (self.alpha * (reward - q))
        return updated_q
