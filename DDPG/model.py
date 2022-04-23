import torch
import torch.nn as nn


class ActorModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorModel, self).__init__()
        hid_size = 100

        self.fc1 = nn.Linear(obs_dim, hid_size)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(hid_size, act_dim)
        self.activation2 = nn.Tanh()

    def forward(self, obs):
        out = self.fc1(obs)
        out = self.activation1(out)
        out = self.fc2(out)
        act = self.activation2(out)
        return act


class CriticModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(CriticModel, self).__init__()

        hid_size = 100
        input_dim = obs_dim + act_dim

        self.fc1 = nn.Linear(input_dim, hid_size)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(hid_size, 1)

    def forward(self, obs, act):
        concat = torch.cat((obs, act), dim=1)
        out = self.fc1(concat)
        out = self.activation1(out)
        Q = self.fc2(out)
        return Q


class Model(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        self.actor_model = ActorModel(obs_dim, act_dim)
        self.critic_model = CriticModel(obs_dim, act_dim)

    def policy(self, obs):
        return self.actor_model.forward(obs)

    def value(self, obs, act):
        return self.critic_model.forward(obs, act)

    # def get_actor_params(self):
    #     return self.actor_model.parameters()  # return the name of parameters
