import os
import numpy as np
import torch


class Agent(object):
    def __init__(self, algorithm, obs_dim, act_dim):
        self.alg = algorithm
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def sample(self, obs):
        obs = np.expand_dims(obs, axis=0)  # expand a dim for batch training
        obs = torch.tensor(obs, dtype=torch.float)
        act_prob = self.alg.predict(obs)
        act_prob = act_prob.detach().numpy()
        act_prob = np.squeeze(act_prob)
        act = np.random.choice(range(self.act_dim), p=act_prob)  # choose the act according to probability
        return act

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = torch.tensor(obs, dtype=torch.float)
        act_prob = self.alg.predict(obs)
        act_prob = act_prob.detach().numpy()
        act_prob = np.squeeze(act_prob)

        act = np.argmax(act_prob)  # choose the action with highest probability
        return act

    def learn(self, obs, act, reward):
        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        obs = torch.tensor(obs, dtype=torch.float)
        loss = self.alg.learn(obs, act, reward)
        return loss

    def save(self, save_path, model=None):
        """Save parameters.

        Args:
            save_path(str): where to save the parameters.
            model(nn.Module): model that describes the neural network structure. If None, will use self.alg.model.

        Raises:
            ValueError: if model is None and self.alg.model does not exist.
        """
        if model is None:
            model = self.alg.model

        torch.save(model.state_dict(), save_path)

