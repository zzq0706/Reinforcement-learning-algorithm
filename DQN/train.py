import gym
import numpy as np
from parl.utils import logger  # , ReplayMemory
from replay_memory import ReplayMemory
from model import Model
from agent import Agent
# from parl.algorithms import DQN
from algorithm import DQNModel


MEMORY_SIZE = 200000
MEMORY_WARMUP_SIZE = 200
GAMMA = 0.99
LEARNING_RATE = 0.0005
LEARN_FREQ = 5
BATCH_SIZE = 64


def run_episode(env, agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)
        next_obs, reward, done, _ = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))
        # rpm.append(obs, action, reward, next_obs, done)

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done) = rpm.sample_batch(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


def evaluate(agent, env, eval_episodes=5, render=False):
    eval_reward = []
    for i in range(eval_episodes):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break

        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

# def evaluate(agent, env, eval_episodes=5, render=False):
#     eval_reward = []
#     for i in range(eval_episodes):
#         obs = env.reset()
#         episode_reward = 0
#         while True:
#             action = agent.predict(obs)
#             obs, reward, done, _ = env.step(action)
#             episode_reward += reward
#             if render:
#                 env.render()
#             if done:
#                 break
#         eval_reward.append(episode_reward)
#     return np.mean(eval_reward)


def main():
    env = gym.make('CartPole-v0')
    action_dim = env.action_space.n
    obs_shape = env.observation_space.shape[0]

    # rpm = ReplayMemory(MEMORY_SIZE, obs_shape, 0)
    rpm = ReplayMemory(MEMORY_SIZE)

    # construct agent
    model = Model(obs_dim=obs_shape, act_dim=action_dim)
    alg = DQNModel(model, gamma=GAMMA, lr=LEARNING_RATE)
    # algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        alg,
        act_dim=action_dim,
        e_greedy=0.1,
        e_greedy_decrement=1e-6  # the exploration slows down with steps increasing
    )

    # load model
    # save_path = './dqn_model.ckpt'
    # agent.restore(save_path)

    # warmup memory
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(env, agent, rpm)

    max_episode = 800

    # start train
    episode = 0
    while episode < max_episode:
        # train part
        for i in range(50):
            total_reward = run_episode(env, agent, rpm)
            episode += 1

        # test part
        eval_reward = evaluate(agent, env, render=False)
        logger.info('episode:{}  e_greedy:{}, test_reward:{}'.format(episode, agent.e_greedy, eval_reward))

    # save model
    # save_path = './dqn_model.ckpt'
    # agent.save(save_path)


if __name__ == '__main__':
    main()
