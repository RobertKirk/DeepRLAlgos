# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Imports #

# %load_ext autoreload
# %autoreload 2

# +
import time

import gym
import math
import random
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# +
random.seed(100)
env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
plt.ion()

device = torch.device("cuda")


# -

class VPGNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dims=[32], discrete_action=True):
        super(VPGNetwork, self).__init__()
        self.discrete_action = discrete_action
        self.layers = [nn.Linear(observation_dim, hidden_dims[0])]
        self.add_module(f"layer{1}", self.layers[0])
        for i in range(0, len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.add_module(f"layer{len(self.layers)}", self.layers[-1])
        self.layers.append(nn.Linear(hidden_dims[-1], action_dim))
        self.add_module(f"layer{len(self.layers)}", self.layers[-1])
    
    def _action_calculation(self, observation):
        if self.discrete_action:
            return F.softmax(observation, dim=0)
        else:
            return observation
    
    def forward(self, observation):
        for layer in self.layers:
            observation = F.relu(layer(observation))
        return self._action_calculation(observation)


# +
BATCH_SIZE = 5000
OBSERVATION_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

policy = VPGNetwork(OBSERVATION_DIM, ACTION_DIM, [32, 32], True)

optimizer = optim.RMSprop(policy.parameters(), lr=0.005)
# -

list(policy.modules())


def choose_action(obs):
    act_probs = policy(torch.as_tensor(obs).float())
    act_dist = torch.distributions.Multinomial(1, probs=act_probs)
    sample = act_dist.sample()
    return sample.max(0)[1].item(), act_probs


def act(n_episodes):
    for _ in range(0, n_episodes):
        obs = env.reset()
        done = False
        while not done:
            env.render()
            action, _ = choose_action(obs)
            obs, _, done, _ = env.step(action)
        time.sleep(0.01)
    env.close()


def do_one_epoch():
    trajectories_rewards = []
    trajectories_acts = []
    trajectories_act_probs = []
    trajectories_obs = []
    while True:
        # rendering = i == 1
        obs = env.reset()
        done = False
        episode_len = 0
        while not done:
            action, act_probs = choose_action(obs)
            trajectories_act_probs.append(act_probs)
            trajectories_acts.append([action])
            trajectories_obs.append(obs)
            episode_len += 1
            obs, _, done, _ = env.step(action)
        trajectories_rewards += reversed(range(0, episode_len))
        
        if len(trajectories_obs) > BATCH_SIZE:
            break

    rews = torch.as_tensor(trajectories_rewards).float()
    act_probs = torch.stack(trajectories_act_probs).float()
    acts = torch.as_tensor(trajectories_acts).long()
    obs = torch.as_tensor(trajectories_obs).float()

    log_probs = (torch.zeros(act_probs.shape).scatter_(1, acts, 1) * torch.log(act_probs)).sum(dim=1)
    loss = -torch.mean(log_probs * rews)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    plot_grad_flow(policy.named_parameters())

    avg_reward = torch.mean(rews)
    return avg_reward


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            mplt.lines.Line2D([0], [0], color="c", lw=4),
            mplt.lines.Line2D([0], [0], color="b", lw=4),
            mplt.lines.Line2D([0], [0], color="k", lw=4)
        ],
        ['max-gradient', 'mean-gradient', 'zero-gradient']
    )
    plt.show()


def train(epochs):
    epoch_avg_rewards = []
    for i in range(0, epochs):
        avg_rew = do_one_epoch()
        epoch_avg_rewards.append(avg_rew)
        print(f"epoch{i} done, avg_reward: {avg_rew}")
    plt.plot(epoch_avg_rewards)


act(4)

train(50)
