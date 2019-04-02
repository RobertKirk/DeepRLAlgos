import torch.nn as nn
import torch.nn.functional as F


class LinearNetwork(nn.Module):
    def __init__(
        self, observation_dim, action_dim, hidden_dims=[32], discrete_action=True
    ):
        super(LinearNetwork, self).__init__()
        self.discrete_action = discrete_action
        self.layers = nn.ModuleList([nn.Linear(observation_dim, hidden_dims[0])])
        for i in range(0, len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.layers.append(nn.Linear(hidden_dims[-1], action_dim))

    def _action_calculation(self, observation):
        if self.discrete_action:
            return F.softmax(observation, dim=0)
        else:
            return observation

    def forward(self, observation):
        for layer in self.layers:
            observation = F.relu(layer(observation))
        return self._action_calculation(observation)
