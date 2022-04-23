import torch.nn as nn
# import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        hid1_size = 128
        hid2_size = 128
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.activation2 = nn.ReLU()
        self.fc3 = nn.Linear(hid2_size, act_dim)

    def forward(self, obs):
        obs = self.fc1(obs)
        obs = self.activation1(obs)
        obs = self.fc2(obs)
        obs = self.activation2(obs)
        q = self.fc3(obs)
        return q
