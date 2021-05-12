import numpy as np
import copy

FILED_TIPE = {
    "N":0, 
    "G":2, 
    }

ACTIONS = {
    "UP": 0, 
    "DOWN": 1, 
    "LEFT": 2, 
    "RIGHT":3
    }

class GSD():
    """
        Gaussian Squeeze Domain(GSD)
    """
    def __init__(self, mu=175.0, sigma=175.0, actions=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
        self.mu = mu
        self.sigma = sigma
        self.actions = actions

    def step(self, actions):
        reward = self._compute_reward(actions)
        return "", reward


    def _compute_reward(self, actions):
        x = np.sum(actions)
        reward = x * np.exp(-(x-self.mu)**2/self.sigma**2)
        return reward
