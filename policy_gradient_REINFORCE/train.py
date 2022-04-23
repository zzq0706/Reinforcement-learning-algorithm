import gym
import numpy as np
from model import PGModel
from agent import Agent
from algorithm import PolicyGradient
from parl.utils import logger


LEARNING_RATE = 0.001
GAMMA = 1.0


def calc_reward_to_go(reward_list, gamma=GAMMA):
    """
    calculate the reward considering the future steps
    """
    for i in range(len(reward_list) - 2, -1, -1):
        # G_t = r_t + gamma * r_t+1 + ... = r_t + gamma * G_t+1
        reward_list[i] += gamma * reward_list[i + 1]  # G_t
    return np.array(reward_list)


def run_episode(env, agent):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)

        obs, reward, done, info = env.step(action)
        reward_list.append(reward)

        if done:
            break

    return obs_list, action_list, reward_list


def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def main():
    env = gym.make("CartPole-v0")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = PGModel(obs_dim=obs_dim, act_dim=act_dim)
    alg = PolicyGradient(model, lr=LEARNING_RATE)
    agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim)

    for i in range(100):
        obs_list, action_list, reward_list = run_episode(env, agent)
        if i % 10 == 0:
            logger.info("Episode {}, Reward sum {}.".format(i, sum(reward_list)))

        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = calc_reward_to_go(reward_list)

        agent.learn(batch_obs, batch_action, batch_reward)
        if (i + 1) % 100 == 0:
            total_reward = evaluate(env, agent, render=True)
            logger.info('Test reward: {}'.format(total_reward))

    save_path = 'policy_gradient_model.pth'
    agent.save(save_path)

if __name__ == '__main__':
    main()
