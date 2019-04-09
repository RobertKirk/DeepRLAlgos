from abc import ABC
import random

import gym
import torch


class RLAlgorithm(ABC):
    def __init__(self, env, random_seed, *args, **kwargs):
        self.env = gym.make(env)
        self.uw_env = gym.make(env).unwrapped
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n
        self.random_seed = random_seed
        random.seed(self.random_seed)
        self.device = torch.device("cuda")

    def train(self, lr, batch_size, epochs):
        pass

    def loss_function(self, *args, **kwargs):
        pass

    def choose_action(self, obs):
        pass

