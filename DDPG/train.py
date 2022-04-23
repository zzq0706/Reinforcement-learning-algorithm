import torch
import numpy as np
from parl.utils import logger
from replay_memory import ReplayMemory
from model import Model, ActorModel, CriticModel
from agent import Agent
from algorithm import DDPG
from env import ContinuousCartPoleEnv


NOISE = 0.05
REWARD_SCALE = 0.1
BATCH_SIZE = 128
GAMMA = 0.99
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
MEMORY_SIZE = int(1e6)
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 20


def run_train_episode(agent, env, rpm):
    obs = env.reset()
    total_reward = 0
    steps = 0
    while True:
        steps += 1
        # batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(obs)

        action = np.clip(np.random.normal(action, NOISE), -1.0, 1.0)

        action = np.asarray(action, dtype='float32')

        # action_noise = np.random.normal(0, NOISE)
        # action = (action + action_noise).clip(-1, 1)
        # action = np.array(action)
        # action = action.view(dtype='float64')

        next_obs, reward, done, info = env.step(action)

        action = [action]
        rpm.append((obs, action, REWARD_SCALE * reward, next_obs, done))

        if len(rpm) > MEMORY_WARMUP_SIZE and (steps % 5) == 0:
            (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done) = rpm.sample_batch(BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)

        obs = next_obs
        total_reward += reward

        if done or steps >= 200:
            break

    return total_reward


def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward = 0
        steps = 0
        while True:
            action = agent.predict(obs)
            action = np.clip(action, -1.0, 1.0)
            steps += 1
            action = np.asarray(action, dtype='float32')
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            total_reward += reward

            if render:
                env.render()
            if done or steps >= 200:
                break
        eval_reward.append(total_reward)

    return np.mean(eval_reward)


def main():
    env = ContinuousCartPoleEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = Model(obs_dim, act_dim)
    alg = DDPG(model, gamma=GAMMA, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = Agent(alg, obs_dim, act_dim)

    rpm = ReplayMemory(MEMORY_SIZE)

    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_train_episode(agent, env, rpm)

    episode = 0
    while episode < 5000:
        for i in range(50):
            total_reward = run_train_episode(agent, env,rpm)
            episode += 1

        eval_reward = evaluate(env, agent, render=False)
        logger.info('episode:{}    Test reward:{}'.format(episode, eval_reward))


if __name__ == '__main__':
    main()