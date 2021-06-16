import math
from itertools import count
import pickle
import numpy as np
import os

import gym
from pyrep.errors import ConfigurationError, ConfigurationPathError

from utils.zfilter import ZFilter


# def expert_perfect_example(n_episodes: int, ep_length: int):
#     expert_env = gym.make("gym_reach:reachPerfectExp-v0")
#     expert_env.reset()
#     for episode in range(n_episodes):
#         obs = expert_env.reset()
#         ik_path = expert_env.panda.get_path(position=expert_env.target.get_position(), euler=[0, math.radians(180), 0])
#         path_done = False
#         rewards = []
#         for i in range(ep_length):
#             if not path_done:
#                 path_done = ik_path.step()
#             obs, reward, done, info = expert_env.step(np.zeros(7))
#             # print("reward: ", reward)
#             rewards.append(reward)
#             if done:
#                 print("Episode finished early after {} timesteps".format(i + 1))
#                 print("Total Reward: ", np.sum(rewards))
#                 break
#
#     expert_env.close()

def run(max_expert_state_num: int):
    num_steps = 0

    env_name = "gym_reach:reachNoisy-v0"
    env = gym.make(env_name, control_loop_enabled=True)
    expert_traj = []
    state_only_expert_traj = []

    state_dim = env.observation_space.shape[0]
    running_state = ZFilter((state_dim,), clip=5)

    for i_episode in count():

        state = env.reset()
        state = running_state(state)
        print(state.shape)
        reward_episode = 0

        try:
            ik_path = env.panda.get_path(position=env.target.get_position(), euler=[0, math.radians(180), 0])
        except ConfigurationPathError as e:
            print("No path", e)
            continue

        path_done = False

        for t in range(10000):
            if not path_done:
                path_done = ik_path.step()
                # action = np.array(env.panda.get_joint_target_velocities()).astype(np.float64)

            next_state, reward, done, _ = env.step(np.zeros(7))
            action = np.array(env.panda.get_joint_velocities()).astype(np.float64)
            # print("Action: ", action)
            next_state = running_state(next_state)

            reward_episode += reward
            num_steps += 1

            expert_traj.append(np.hstack([state, action]))
            state_only_expert_traj.append(np.hstack([state, next_state]))

            if path_done or done or num_steps >= max_expert_state_num:
                break

            state = next_state

        print('Episode {}\t reward: {:.2f}'.format(i_episode, reward_episode))

        if num_steps >= max_expert_state_num:
            break

    expert_traj = np.stack(expert_traj)
    state_only_expert_traj = np.stack(state_only_expert_traj)
    pickle.dump((expert_traj, running_state), open('expert_traj/{}_expert_traj.p'.format(env_name), 'wb'))
    pickle.dump((state_only_expert_traj, running_state), open('expert_traj/{}_state_only_expert_traj.p'.format(env_name), 'wb'))


if __name__ == "__main__":
    run(10000)
