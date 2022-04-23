import torch


class PolicyGradient(object):
    def __init__(self, model, lr=None):
        self.model = model
        isinstance(lr, float)
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def predict(self, obs):
        act = self.model(obs)
        return act

    def learn(self, obs, action, reward):
        act_prob = self.model(obs)
        # cross entropy e.g -1 * log([0.2 0.5 0.3] * [0 1 0]
        reward = torch.tensor(reward, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        action = torch.squeeze(action)
        one_hot = torch.nn.functional.one_hot(action, num_classes=act_prob.shape[1])
        log_prob = torch.sum(-1.0 * torch.log(act_prob) * one_hot, dim=1, keepdim=True)
        loss = log_prob * reward
        loss = torch.mean(loss)

        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().numpy()
