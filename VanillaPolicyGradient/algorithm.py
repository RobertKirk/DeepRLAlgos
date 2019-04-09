import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from common.networks import LinearNetwork
from common.algorithm import RLAlgorithm

plt.ion()


class VanillaPolicyGradient(RLAlgorithm):
    def __init__(self, env, random_seed, hidden_dims=[32, 32], learning_rate=0.005):
        super().__init__(env, random_seed)
        self.policy_net = LinearNetwork(self.obs_dim, self.act_dim, hidden_dims, True).to(self.device)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=learning_rate)

    def _do_one_epoch(self, batch_size):
        trajectories_rewards = []
        trajectories_acts = []
        trajectories_act_probs = []
        trajectories_obs = []
        while True:
            # import pdb; pdb.set_trace()
            obs = self.env.reset()
            done = False
            episode_len = 0
            episode_rewards = []
            while not done:
                action, act_probs = self.choose_action(obs)
                trajectories_act_probs.append(act_probs)
                trajectories_acts.append([action])
                trajectories_obs.append(obs)
                episode_len += 1
                obs, reward, done, _ = self.env.step(action)
                episode_rewards.append(reward)

            trajectories_rewards += self._reward_to_go(episode_rewards)
            if len(trajectories_obs) > batch_size:
                break

        act_probs_tensor = torch.stack(trajectories_act_probs).to(self.device)
        act_tensor = torch.as_tensor(trajectories_acts).to(self.device)
        rewards_tensor = torch.as_tensor(trajectories_rewards).to(self.device)

        loss = self.loss_function(act_probs_tensor, act_tensor, rewards_tensor)
        # print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        avg_reward = np.mean(trajectories_rewards)
        return avg_reward

    def train(self, batch_size, epochs, lr=None):
        if lr:
            self.set_learning_rate(lr)
        epoch_avg_rewards = []
        for i in range(0, epochs):
            avg_rew = self._do_one_epoch(batch_size)
            epoch_avg_rewards.append(avg_rew)
            print(f"epoch{i} done, avg_reward: {avg_rew}")

    def set_learning_rate(self, lr):
        self.optimizer.defaults['lr'] = lr

    def choose_action(self, obs):
        act_probs = self.policy_net(torch.as_tensor(obs).float().to(self.device))
        act_dist = torch.distributions.Multinomial(1, probs=act_probs)
        sample = act_dist.sample().to(self.device)
        return sample.max(0)[1].item(), act_probs

    def _reward_to_go(self, episode_rewards):
        episode_len = len(episode_rewards)
        rewards_to_go = [episode_rewards[-1]]
        for i in range(1, episode_len):
            rewards_to_go.append(episode_rewards[-(i + 1)] + rewards_to_go[i - 1])
        rewards_to_go.reverse()
        print(rewards_to_go)
        return rewards_to_go

    def loss_function(self, act_probs, acts, rews):
        log_probs = (
            torch.cuda.FloatTensor(act_probs.shape).fill_(0).scatter_(1, acts, 1)
            * torch.log(act_probs)
        ).sum(dim=1)
        return -torch.mean(log_probs * rews)

    def act(self, n_episodes):
        try:
            for _ in range(0, n_episodes):
                obs = self.uw_env.reset()
                done = False
                while not done:
                    self.uw_env.render()
                    action, _ = self.choose_action(obs)
                    obs, _, done, _ = self.uw_env.step(action)
                time.sleep(0.01)
            self.uw_env.close()
        except Exception:
            self.uw_env.close()
            raise

