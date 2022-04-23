import numpy as np
import torch


class Agent(object):
    def __init__(self, algorithm, obs_dim, act_dim):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # super(Agent, self).__init__(algorithm)
        self.alg = algorithm
        self.alg.sync_target(decay=0)

    def learn(self, obs, act, reward, next_obs, terminal):
        # obs = np.expand_dims(obs, axis=0)
        # act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        terminal = np.expand_dims(terminal, axis=-1)
        obs = torch.tensor(obs, dtype=torch.float)
        act = torch.tensor(act, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_obs = torch.tensor(next_obs, dtype=torch.float)
        terminal = torch.tensor(terminal, dtype=torch.float)
        actor_loss, critic_loss = self.alg.learn(obs, act, reward, next_obs, terminal)
        self.alg.sync_target()
        return actor_loss.detach().numpy(), critic_loss.detach().numpy()

    def predict(self, obs):
        """
        predict an action when given an observation
        :param obs: list with dim (4, )
        :return: action
        """
        # obs = np.expand_dims(obs, axis=0)
        obs = torch.tensor(obs, dtype=torch.float)
        act = self.alg.predict(obs)
        act = act.detach().numpy()
        act = np.squeeze(act)

        return act
