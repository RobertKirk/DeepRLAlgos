import gym
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from common.networks import LinearNetwork

plt.ion()

device = torch.device("cuda")


def choose_action(policy_net, obs):
    act_probs = policy_net(torch.as_tensor(obs).float().to(device))
    act_dist = torch.distributions.Multinomial(1, probs=act_probs)
    sample = act_dist.sample().to(device)
    return sample.max(0)[1].item(), act_probs


def reward_to_go(episode_rewards):
    episode_len = len(episode_rewards)
    rewards_to_go = [episode_rewards[-1]]
    for i in range(1, episode_len):
        rewards_to_go.append(episode_rewards[-(i + 1)] + rewards_to_go[i - 1])
    rewards_to_go.reverse()
    return rewards_to_go


def loss_function(act_probs, acts, rews):
    log_probs = (
        torch.cuda.FloatTensor(act_probs.shape).fill_(0).scatter_(1, acts, 1)
        * torch.log(act_probs)
    ).sum(dim=1)
    return -torch.mean(log_probs * rews)


def do_one_epoch(env, policy_net, optimizer, batch_size):
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
            action, act_probs = choose_action(policy_net, obs)
            trajectories_act_probs.append(act_probs)
            trajectories_acts.append([action])
            trajectories_obs.append(obs)
            episode_len += 1
            obs, reward, done, _ = env.step(action)
            episode_rewards.append(reward)

        trajectories_rewards += reward_to_go(episode_rewards)

        if len(trajectories_obs) > batch_size:
            break

    act_probs_tensor = torch.stack(trajectories_act_probs).to(device)
    act_tensor = torch.as_tensor(trajectories_acts).to(device)
    rewards_tensor = torch.as_tensor(trajectories_rewards).to(device)

    loss = loss_function(act_probs_tensor, act_tensor, rewards_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_reward = np.mean(trajectories_rewards)
    return avg_reward


def vanilla_policy_gradient(
    epochs=50, env="CartPole-v0", batch_size=5000, random_seed=100, learning_rate=0.005
):
    env = gym.make(env).unwrapped
    random.seed(random_seed)

    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = LinearNetwork(observation_dim, action_dim, [32, 32], True).to(device)

    optimizer = optim.RMSprop(policy_net.parameters(), lr=learning_rate)

    epoch_avg_rewards = []

    for i in range(0, epochs):
        avg_rew = do_one_epoch(env, policy_net, optimizer, batch_size)
        epoch_avg_rewards.append(avg_rew)
        print(f"epoch{i} done, avg_reward: {avg_rew}")

    plt.plot(epoch_avg_rewards)
    return policy_net
