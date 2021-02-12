from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.const import PrimitiveShape
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
import numpy as np
import math
import sys
import argparse


class NoisyReacherEnv(object):
    def __init__(self, action_noise_mean=0.0, action_noise_var=0.0, control_loop_enabled=False, headless=False):
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=headless)
        self.pr.start()
        self.agent = Panda()
        self.agent.set_control_loop_enabled(control_loop_enabled)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.target = Shape.create(type=PrimitiveShape.SPHERE,
                      size=[0.05, 0.05, 0.05],
                      color=[1.0, 0.1, 0.1],
                      static=True, respondable=False)
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()

        self.action_noise_mean = action_noise_mean
        self.action_noise_variance = action_noise_var

    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position
        return np.concatenate([self.agent.get_joint_positions(),
                               self.agent.get_joint_velocities(),
                               self.target.get_position()])

    def reward(self):
        ax, ay, az = self.agent_ee_tip.get_position()
        tx, ty, tz = self.target.get_position()
        # Reward is negative distance to target
        return -np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)


    def reset(self):
        # Get a random position within a cuboid and set the target position
        pos = list(np.random.uniform(POS_MIN, POS_MAX))
        self.target.set_position(pos)
        self.agent.set_joint_positions(self.initial_joint_positions, True)
        return self._get_state()

    def step(self, action: np.array):
        noise = np.random.normal(self.action_noise_mean, self.action_noise_variance, action.size)
        self.agent.set_joint_target_velocities(action + noise)  # Execute action on arm, with actuator noise
        # self.agent.set_joint_target_velocities([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.pr.step()  # Step the physics simulation
        return self.reward(), self._get_state()

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()


class Agent(object):

    def act(self, state) -> np.array:
        del state
        return np.random.uniform(-1.0, 1.0, size=(7,))

    def learn(self, replay_buffer):
        del replay_buffer
        pass


if __name__ == "__main__":
    SCENE_FILE = join(dirname(abspath(__file__)), 'scene_panda_reach_target.ttt')
    POS_MIN, POS_MAX = [0.8, -0.2, 1.0], [1.0, 0.2, 1.4]

    parser = argparse.ArgumentParser()
    parser.add_argument("agent_type", help="Type of robotic controller: 'agent' or 'expert'", type=str)
    parser.add_argument("episode_length", help="number of steps in each episode", type=int)
    parser.add_argument("num_episodes", help="the total number of episodes before termination", type=int)
    parser.add_argument("--headless", help="Run simulations in headless mode (without gui)", action="store_true")
    args = parser.parse_args()

    AGENT_TYPE = args.agent_type
    EPISODE_LENGTH = args.episode_length
    EPISODES = args.num_episodes
    HEADLESS = args.headless


    # Example Running with an Expert Demonstration
    if AGENT_TYPE == 'expert':
        expert_env = NoisyReacherEnv(0.0, 0.0, control_loop_enabled=True, headless=HEADLESS)  # Lets have the expert run without noise
        for e in range(EPISODES):
            print('Starting episode %d' % e)
            state = expert_env.reset()
            print(expert_env.target.get_position())
            target_pos = list(expert_env.target.get_position())
            path = expert_env.agent.get_path( # Expert just uses Inverse Kinematics to choose actions
                position=target_pos, euler=[0, math.radians(180), 0])
            done = False
            for i in range(EPISODE_LENGTH):
                if not done:
                    done = path.step()
                expert_env.pr.step()
                reward, next_state = expert_env.reward(), expert_env._get_state
                state = next_state

        print("Done!")
        expert_env.shutdown()

    # Example Running with an RL Agent in a noisy environment
    elif AGENT_TYPE == 'agent':
        env = NoisyReacherEnv(0.1, 0.05, headless=HEADLESS)
        agent = Agent()
        replay_buffer = []

        for e in range(EPISODES):

            print('Starting episode %d' % e)
            state = env.reset()
            for i in range(EPISODE_LENGTH):
                action = agent.act(state)
                reward, next_state = env.step(action)
                replay_buffer.append((state, action, reward, next_state))
                state = next_state
                agent.learn(replay_buffer)

        print('Done!')
        env.shutdown()
