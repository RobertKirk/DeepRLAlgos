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

from common.networks import LinearNetwork

# +
random.seed(100)
env = gym.make("CartPole-v0").unwrapped

# set up matplotlib
plt.ion()

device = torch.device("cuda")


# -


# +
BATCH_SIZE = 5000
OBSERVATION_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

policy = LinearNetwork(OBSERVATION_DIM, ACTION_DIM, [32, 32], True)

optimizer = optim.RMSprop(policy.parameters(), lr=0.005)


# -


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


def loss_function(act_probs, acts, rews):
    log_probs = (
        torch.zeros(act_probs.shape).scatter_(1, acts, 1) * torch.log(act_probs)
    ).sum(dim=1)
    return -torch.mean(log_probs * rews)


def reward_to_go(episode_rewards):
    episode_len = len(episode_rewards)
    rewards_to_go = [episode_rewards[-1]]
    for i in range(1, episode_len):
        rewards_to_go[i] = episode_rewards[-(i + 1)] + rewards_to_go[i - 1]
    return reverse(rewards_to_go)


def do_one_epoch():
    trajectories_rewards = []
    trajectories_acts = []
    trajectories_act_probs = []
    trajectories_obs = []
    while True:
        obs = env.reset()
        done = False
        episode_len = 0
        episode_rewards = []
        while not done:
            action, act_probs = choose_action(obs)
            trajectories_act_probs.append(act_probs)
            trajectories_acts.append([action])
            trajectories_obs.append(obs)
            episode_len += 1
            obs, reward, done, _ = env.step(action)
            episode_rewards.append(reward)

        trajectories_rewards += reward_to_go(episode_rewards)

        if len(trajectories_obs) > BATCH_SIZE:
            break

    loss = loss_function(
        torch.stack(trajectories_act_probs).float(),
        torch.as_tensor(trajectories_acts).long(),
        torch.as_tensor(trajectories_rewards).float(),
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_reward = np.mean(trajectories_rewards)
    return avg_reward


def train(epochs):
    epoch_avg_rewards = []
    for i in range(0, epochs):
        avg_rew = do_one_epoch()
        epoch_avg_rewards.append(avg_rew)
        print(f"epoch{i} done, avg_reward: {avg_rew}")
        if 1 % 10 == 0:
            print(gc.collect)
    plt.plot(epoch_avg_rewards)


train(50)
