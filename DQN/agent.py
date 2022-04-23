import numpy as np
# import parl
import torch


class Agent(object):  # parl.Agent
    def __init__(self, algorithm, act_dim, e_greedy=0.1, e_greedy_decrement=0):
        # assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.act_dim = act_dim

        # super(Agent, self).__init__(algorithm)
        self.alg = algorithm

        self.global_step = 0
        self.update_target_steps = 200  # every 200 steps copy the paramters of mdodel to target_model
        self.e_greedy = e_greedy
        self.e_greedy_decrement = e_greedy_decrement

    def learn(self, obs, act, reward, next_obs, terminal):
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        terminal = np.expand_dims(terminal, axis=-1)
        obs = torch.tensor(obs, dtype=torch.float)
        act = torch.tensor(act, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_obs = torch.tensor(next_obs, dtype=torch.float)
        terminal = torch.tensor(terminal, dtype=torch.float)
        loss = self.alg.learn(obs, act, reward, next_obs, terminal)
        return loss.detach().numpy()

    def sample(self, obs):
        """
        sample an action with exploration when given an observation
        :param obs:
        :return:
        """
        sample = np.random.random()  # generate a number between 0, 1
        if sample < self.e_greedy:
            act = np.random.randint(self.act_dim)
        else:
            if np.random.random() < 0.01:
                act = np.random.randint(self.act_dim)
            else:
                act = self.predict(obs)
        self.e_greedy = max(0.01, self.e_greedy - self.e_greedy_decrement)
        return act

    def predict(self, obs):
        """
        predict an action when given an observation
        :param obs:
        :return:
        """
        obs = torch.tensor(obs, dtype=torch.float)
        pred_Q = self.alg.predict(obs)
        act = pred_Q.argmax().numpy()
        return act
