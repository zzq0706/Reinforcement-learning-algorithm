import gym
from gridworld import CliffWalkingWrapper
from agent import QAgent


def run_episode(env, agent, render=False):
    total_steps = 0
    total_reward = 0
    obs = env.reset()  # start a new episode
    action = agent.sample(obs)

    while True:
        next_obs, reward, done, info = env.step(action)  # interact one time with env
        next_action = agent.sample(next_obs)

        # update Q-table
        agent.learn(obs, action, reward, next_obs, done)

        action = next_action
        obs = next_obs

        total_reward += reward
        total_steps += 1
        if render:
            env.render()  # render a new picture
        if done:
            break  # break the loop when the training is finished

    return total_reward, total_steps


def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs)  # greedy
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        obs = next_obs
        env.render()
        if done:
            print('test reward: %.1f' % total_reward)
            break


def main():
    env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
    env = CliffWalkingWrapper(env)

    agent = QAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        epsilon=0.1,
        learning_rate=0.1,
        gamma=0.9)

    is_render = False

    for episode in range(500):
        ep_reward, ep_steps = run_episode(env, agent, is_render)

        if episode % 20 == 0:
            is_render = True
            print('Episode %s: steps = %s, reward = %.1f' % (episode, ep_steps, ep_reward))
        else:
            is_render = False

    test_episode(env, agent)


if __name__ == '__main__':
    main()
