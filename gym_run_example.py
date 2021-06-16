import argparse
import math
from os.path import join

import gym
import numpy as np


class Agent(object):
    def act(self, obs) -> np.array:
        return np.random.uniform(-1.0, 1.0, size=(7,))

    def learn(self, history):
        pass


## Example of running in noisy agent environment
def noisy_agent_example(n_episodes: int, ep_length: int):
    noisy_agent_env = gym.make("gym_reach:reachNoisy-v0", render_mode='rgb_array')
    noisy_agent_env.reset()
    a = Agent()
    for episode in range(n_episodes):
        obs = noisy_agent_env.reset()
        for i in range(ep_length):
            action = a.act(obs)
            im = noisy_agent_env.render('rgb_array')
            obs, reward, done, info = noisy_agent_env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(i + 1))
                break
    noisy_agent_env.close()


## Example of running in perfect expert environment (with demonstrations created by IK_Solver)
def expert_perfect_example(n_episodes: int, ep_length: int):
    expert_env = gym.make("gym_reach:reachPerfectExp-v0")
    expert_env.reset()
    for episode in range(n_episodes):
        obs = expert_env.reset()
        ik_path = expert_env.panda.get_path(position=expert_env.target.get_position(), euler=[0, math.radians(180), 0])
        path_done = False
        rewards = []
        for i in range(ep_length):
            if not path_done:
                path_done = ik_path.step()
            obs, reward, done, info = expert_env.step(np.zeros(7))
            # print("reward: ", reward)
            rewards.append(reward)
            if done:
                print("Episode finished early after {} timesteps".format(i + 1))
                print("Total Reward: ", np.sum(rewards))
                break

    expert_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_type", help="Type of robotic controller: 'agent' or 'expert'", type=str)
    parser.add_argument("episode_length", help="number of steps in each episode", type=int)
    parser.add_argument("num_episodes", help="the total number of episodes before termination", type=int)
    args = parser.parse_args()

    AGENT_TYPE = args.agent_type
    EPISODE_LENGTH = args.episode_length
    EPISODES = args.num_episodes

    if AGENT_TYPE == 'agent':
        noisy_agent_example(EPISODES, EPISODE_LENGTH)

    if AGENT_TYPE == 'expert':
        expert_perfect_example(EPISODES, EPISODE_LENGTH)
