import numpy as np


class QAgent(object):
    def __init__(self, obs_n, act_n, epsilon=0.5, learning_rate=0.1, gamma=0.9):
        self.act_n = act_n  # the optional actions
        self.epsilon = epsilon  # e-greed
        self.lr = learning_rate
        self.gamma = gamma
        self.Q = np.zeros((obs_n, act_n))  # Q-table

    def sample(self, obs):
        """
        according to the observation, output the action, with exploration
        """
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):  # choose action according to Q-table
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)  # randomly choose the action
        return action

    def predict(self, obs):
        """
        according to the observation, output the action, without exploration
        """
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  # return the action list, maxQ may corresponds to multiple action
        action = np.random.choice(action_list)
        return action

    def learn(self, obs, action, reward, next_obs, done):
        """
        on-policy
        :param obs: observation before interacting with env
        :param action: action after interacting
        :param reward: reward after interacting
        :param next_obs: observation after interacting
        :param next_action: the action chosen for the next observation according to the Q-table
        :param done: episode finished or not
        :return:
        """
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward  # no next observation
        else:
            target_Q = reward + self.gamma * np.max(self.Q[next_obs, :])
        self.Q[obs, action] = predict_Q + self.lr * (target_Q - predict_Q)

