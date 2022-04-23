import copy
# import parl
import torch


class DQNModel(object):  # parl.Algorithm
    def __init__(self, model, gamma=None, lr=None):
        """
        DQN
        :param model: model of NN
        :param gamma: decay factor of reward
        :param lr: learning rate
        :return:
        """
        assert isinstance(gamma, float)
        assert isinstance(lr, float)

        self.model = model
        self.target_model = copy.deepcopy(model)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.target_model.to(device)

        self.gamma = gamma
        self.lr = lr

        self.mse_loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def sync_target(self):
        """
        synchronize the parameters of self.model to self.target_model
        """
        target_vars = dict(self.target_model.named_parameters())
        for name, var in self.model.named_parameters():
            target_vars[name].data.copy_(var.data)

    def predict(self, obs):
        """
        get [Q(s, a1), Q(s, a2), ...] using the value of self.model
        """
        pred_q = self.model(obs)
        return pred_q

    def learn(self, obs, action, reward, next_obs, terminal):
        # pred_value = self.model(obs).gather(1, action)
        # with torch.no_grad():
        #     max_v = self.target_model(next_obs).max(1, keepdim=True)[0]
        #     target = reward + (1 - terminal) * self.gamma * max_v
        # self.optimizer.zero_grad()
        # loss = self.mse_loss(pred_value, target)
        # loss.backward()
        # self.optimizer.step()
        # return loss.item()

        pred_values = self.model(obs)
        # print('pred_value:{}'.format(pred_values))
        action_dim = pred_values.shape[-1]
        action = torch.squeeze(action)
        action_onehot = torch.nn.functional.one_hot(action, num_classes=action_dim)
        # print(action_onehot)
        pred_value = pred_values * action_onehot
        pred_value = torch.sum(pred_value, axis=1, keepdim=True)
        # print(pred_value)

        # target Q
        with torch.no_grad():
            max_v = self.target_model(next_obs).max(1, keepdim=True)
            # print(max_v)
            target = reward + (1 - terminal) * self.gamma * max_v[0]
        loss = self.mse_loss(pred_value, target)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
