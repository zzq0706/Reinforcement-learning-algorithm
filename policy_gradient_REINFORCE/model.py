import torch.nn as nn


class PGModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PGModel, self).__init__()
        # obs_dim = obs_dim
        # act_dim = act_dim
        hid1_size = act_dim * 10

        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.activation1 = nn.Tanh()
        self.fc2 = nn.Linear(hid1_size, act_dim)
        self.activation2 = nn.Softmax()

    def forward(self, obs):
        out = self.fc1(obs)
        out = self.activation1(out)
        out = self.fc2(out)
        act = self.activation2(out)
        return act

