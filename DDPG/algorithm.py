import copy
import torch


class DDPG(object):
    def __init__(self, model, gamma=0.99, critic_lr=0.001, actor_lr=0.001, syn_decay=0.001):  # model = Model()
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.gamma = gamma
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.tau = syn_decay
        self.optimizer_critic = torch.optim.Adam(self.model.critic_model.parameters(), lr=self.critic_lr)
        self.optimizer_actor = torch.optim.Adam(self.model.actor_model.parameters(), lr=self.actor_lr)
        self.mse_loss = torch.nn.MSELoss()

    def _critic_learn(self, obs, action, reward, next_obs, terminal):

        next_action = self.target_model.policy(next_obs)
        next_Q = self.target_model.value(next_obs, next_action)
        with torch.no_grad():
            target_Q = reward + (1 - terminal) * self.gamma * next_Q

        pred_Q = self.model.value(obs, action)
        loss = self.mse_loss(pred_Q, target_Q)

        self.optimizer_critic.zero_grad()
        loss.backward()
        self.optimizer_critic.step()
        return loss

    def _actor_learn(self, obs):
        action = self.model.policy(obs)
        Q = self.model.value(obs, action)
        loss = torch.mean(-1.0 * Q)  # loss = - Q_w(s, a)

        self.optimizer_actor.zero_grad()
        loss.backward()
        self.optimizer_actor.step()
        return loss

    def sync_target(self, decay=None):
        if decay is None:
            decay = 1.0 - self.tau
        target_vars = dict(self.target_model.named_parameters())
        for name, var in self.model.named_parameters():
            target_vars[name].data.copy_(decay * target_vars[name].data +
                                         (1 - decay) * var.data)

    def predict(self, obs):
        # use actor model in self.model to predict
        return self.model.policy(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        actor_loss =  self._actor_learn(obs)
        critic_loss = self._critic_learn(obs, action, reward, next_obs, terminal)

        return actor_loss, critic_loss





