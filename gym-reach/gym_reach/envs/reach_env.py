import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from pyrep import PyRep
from pyrep.const import PrimitiveShape
from pyrep.objects import Shape
from pyrep.robots.arms.panda import Panda
from pyrep.robots.arms.dobot import Dobot
from pyrep.objects.vision_sensor import VisionSensor
from typing import Union
import importlib.resources as ir


class ReachEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, render_mode: Union[None, str] = None, action_noise_mean=0.0, action_noise_var=0.0, headless=False, control_loop_enabled=False):
        self.action_noise_mean = action_noise_mean
        self.action_noise_variance = action_noise_var
        self.POS_MIN = np.array([0.8, -0.2, 1.0])
        self.POS_MAX = np.array([1.0, 0.2, 1.4])

        self.pr = PyRep()
        # print("PACKAGE CONTENTS", list(ir.contents('gym_reach.scenes')))
        with ir.path("gym_reach.scenes", "scene_panda_reach_target.ttt") as p:
            self.pr.launch(str(p), headless=headless)

        if render_mode is not None:
            self.camera = VisionSensor.create([512, 512],
                                              position=[2.475, -0.05, 1.9],
                                              orientation=np.array([-180.0, -65.0, 90.0]) * np.pi / 180.0)
            print(self.camera.get_render_mode())
        self.pr.start()
        self.panda = Panda()
        self.panda.set_control_loop_enabled(control_loop_enabled)
        self.panda.set_motor_locked_at_zero_velocity(True)
        self.target = Shape.create(type=PrimitiveShape.SPHERE,
                                   size=[0.05, 0.05, 0.05],
                                   color=[1.0, 0.1, 0.1],
                                   static=True, respondable=False)
        self.panda_ee_tip = self.panda.get_tip()
        self.initial_joint_positions = self.panda.get_joint_positions()

        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(7,))

        self.observation_space = spaces.Box(low=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, *self.POS_MIN]),
                                            high=np.array([2.8973, 1.7628, 2.8973, 0.0698, 2.8973, 3.7525, 2.8973, *self.POS_MAX]))

        # self._config_env()

    def _config_env(self):
        pass

    def reward(self):
        ax, ay, az = self.panda_ee_tip.get_position()
        tx, ty, tz = self.target.get_position()
        # Reward is negative distance to target
        return -np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)

    def is_over(self):
        p_tip = np.array(self.panda_ee_tip.get_position())
        target_pos = np.array(self.target.get_position())

        dist_to_target = np.linalg.norm(p_tip - target_pos)
        return dist_to_target <= 0.05

    def _get_state(self):
        return np.concatenate([self.panda.get_joint_positions(), self.target.get_position()])

    def step(self, action):
        noise = np.random.normal(self.action_noise_mean, self.action_noise_variance, len(action))
        self.panda.set_joint_target_velocities(action + noise)  # Execute action on arm, with actuator noise
        self.pr.step()  # Step the physics simulation
        return self._get_state(), self.reward(), self.is_over(), {}

    def reset(self):
        pos = list(np.random.uniform(self.POS_MIN, self.POS_MAX))
        self.target.set_position(pos)
        self.panda.set_joint_positions(self.initial_joint_positions, True)
        return self._get_state()

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.camera.capture_rgb()

    def close(self):
        self.pr.stop()
        self.pr.shutdown()

class ReachEnvFixed(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, render_mode: Union[None, str] = None, action_noise_mean=0.0, action_noise_var=0.0, headless=False, control_loop_enabled=False):
        self.action_noise_mean = action_noise_mean
        self.action_noise_variance = action_noise_var

        self.pr = PyRep()
        # print("PACKAGE CONTENTS", list(ir.contents('gym_reach.scenes')))
        with ir.path("gym_reach.scenes", "scene_panda_reach_target.ttt") as p:
            self.pr.launch(str(p), headless=headless)

        if render_mode is not None:
            self.camera = VisionSensor.create([512, 512],
                                              position=[2.475, -0.05, 1.9],
                                              orientation=np.array([-180.0, -65.0, 90.0]) * np.pi / 180.0)
            print(self.camera.get_render_mode())
        self.pr.start()
        self.panda = Panda()
        self.panda.set_control_loop_enabled(control_loop_enabled)
        self.panda.set_motor_locked_at_zero_velocity(True)
        self.target = Shape.create(type=PrimitiveShape.SPHERE,
                                   size=[0.05, 0.05, 0.05],
                                   color=[1.0, 0.1, 0.1],
                                   static=True, respondable=False)

        self.target_pos = np.array([0.5, -0.3, 1.1])
        self.target.set_position(self.target_pos)

        self.panda_ee_tip = self.panda.get_tip()
        self.initial_joint_positions = self.panda.get_joint_positions()

        self.action_space = spaces.Box(low=-0.8, high=0.8, shape=(7,))

        self.observation_space = spaces.Box(low=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
                                            high=np.array([2.8973, 1.7628, 2.8973, 0.0698, 2.8973, 3.7525, 2.8973]))


    def reward(self):
        tx, ty, tz = self.target_pos
        px, py, pz = self.panda_ee_tip.get_position()
        r_dist_sq = (tx - px) ** 2 + (ty - py) ** 2 + (tz - pz) ** 2
        return -r_dist_sq

        # target_vels = np.array(self.panda.get_joint_target_velocities())
        # r_vel_penalty = -np.linalg.norm(target_vels)
        #
        # divergence_from_safe_pos = -np.linalg.norm(np.array(self.initial_joint_positions) - np.array(self.panda.get_joint_positions()))
        #
        # return 10 * r_dist + 1.0 * r_vel_penalty + 1.0 * 0.1 * divergence_from_safe_pos
        # return -np.linalg.norm(self.target_pos - np.array(self.panda_ee_tip.get_position()))

    def is_over(self):
        p_tip = np.array(self.panda_ee_tip.get_position())
        dist_to_target = np.linalg.norm(p_tip - self.target_pos)

        return dist_to_target <= 0.05

    def _get_state(self):
        return np.array(self.panda.get_joint_positions())

    def step(self, action):
        noise = np.random.normal(self.action_noise_mean, self.action_noise_variance, len(action))
        self.panda.set_joint_target_velocities(action + noise)  # Execute action on arm, with actuator noise
        self.pr.step()  # Step the physics simulation
        return self._get_state(), self.reward(), self.is_over(), {}

    def reset(self):
        self.panda.set_joint_positions(self.initial_joint_positions, True)
        return self._get_state()

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.camera.capture_rgb()

    def close(self):
        self.pr.stop()
        self.pr.shutdown()


class ReachDobotFixed(gym.Env):
    metadata = {'render.modes': ['rgb_array']}


    def __init__(self, render_mode: Union[None, str] = None, action_noise_mean=0.0, action_noise_var=0.0, headless=False, control_loop_enabled=False):
        self.action_noise_mean = action_noise_mean
        self.action_noise_variance = action_noise_var

        self.pr = PyRep()
        # print("PACKAGE CONTENTS", list(ir.contents('gym_reach.scenes')))
        with ir.path("gym_reach.scenes", "scene_dobot_reach_target.ttt") as p:
            self.pr.launch(str(p), headless=headless)

        if render_mode is not None:
            self.camera = VisionSensor.create([512, 512],
                                              position=[2.475, -0.05, 1.9],
                                              orientation=np.array([-180.0, -65.0, 90.0]) * np.pi / 180.0)
            print(self.camera.get_render_mode())
        self.pr.start()
        self.dobot = Dobot()
        self.dobot.set_control_loop_enabled(control_loop_enabled)
        self.dobot.set_motor_locked_at_zero_velocity(True)
        self.target = Shape.create(type=PrimitiveShape.SPHERE,
                                   size=[0.05, 0.05, 0.05],
                                   color=[1.0, 0.1, 0.1],
                                   static=True, respondable=False)

        self.target_pos = np.array([0.15, 0.05, 0.05])
        self.target.set_position(self.target_pos)

        self.dobot_ee_tip = self.dobot.get_tip()

        self.initial_dobot_state = self.dobot.get_configuration_tree()
        self.initial_joint_positions = self.dobot.get_joint_positions()
        self.initial_pos = self.dobot.get_position()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))

        self.observation_space = spaces.Box(low=np.array([-np.pi / 2.0, 0, -0.174533, -np.pi / 2.0]),
                                            high=np.array([np.pi / 2.0, 1.48353, 1.65806, np.pi / 2.0]))


    def reward(self):
        tx, ty, tz = self.target_pos
        px, py, pz = self.dobot_ee_tip.get_position()
        r_dist_sq = (tx - px) ** 2 + (ty - py) ** 2 + (tz - pz) ** 2
        return -r_dist_sq

    def is_over(self):
        p_tip = np.array(self.dobot_ee_tip.get_position())
        dist_to_target = np.linalg.norm(p_tip - self.target_pos)

        return dist_to_target <= 0.05


    def _get_state(self):
        return np.array(self.dobot.get_joint_positions())


    def step(self, action):
        noise = np.random.normal(self.action_noise_mean, self.action_noise_variance, len(action))
        self.dobot.set_joint_target_velocities(action + noise)  # Execute action on arm, with actuator noise
        self.pr.step()  # Step the physics simulation
        return self._get_state(), self.reward(), self.is_over(), {}


    def reset(self):
        self.pr.set_configuration_tree(self.initial_dobot_state)
        self.dobot.set_joint_positions(self.initial_joint_positions, disable_dynamics=True)
        self.dobot.set_joint_target_velocities(np.zeros(4))

        # Let things settle first...
        for i in range(50):
            self.pr.step()

        return self._get_state()


    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.camera.capture_rgb()


    def close(self):
        self.pr.stop()
        self.pr.shutdown()


class ReachDobotMulti(gym.Env):
    metadata = {'render.modes': ['rgb_array']}


    def __init__(self, render_mode: Union[None, str] = None, action_noise_mean=0.0, action_noise_var=0.0, headless=False, control_loop_enabled=False):
        self.action_noise_mean = action_noise_mean
        self.action_noise_variance = action_noise_var

        self.pr = PyRep()
        # print("PACKAGE CONTENTS", list(ir.contents('gym_reach.scenes')))
        with ir.path("gym_reach.scenes", "scene_dobot_reach_target.ttt") as p:
            self.pr.launch(str(p), headless=headless)

        if render_mode is not None:
            self.camera = VisionSensor.create([512, 512],
                                              position=[2.475, -0.05, 1.9],
                                              orientation=np.array([-180.0, -65.0, 90.0]) * np.pi / 180.0)
            print(self.camera.get_render_mode())
        self.pr.start()
        self.dobot = Dobot()
        self.dobot.set_control_loop_enabled(control_loop_enabled)
        self.dobot.set_motor_locked_at_zero_velocity(True)
        self.target = Shape.create(type=PrimitiveShape.SPHERE,
                                   size=[0.05, 0.05, 0.05],
                                   color=[1.0, 0.1, 0.1],
                                   static=True, respondable=False)

        self.target_pos = np.array([0.05, -0.15, 0.10])
        self.target.set_position(self.target_pos)

        self.dobot_ee_tip = self.dobot.get_tip()

        self.initial_dobot_state = self.dobot.get_configuration_tree()
        self.initial_joint_positions = self.dobot.get_joint_positions()
        self.initial_pos = self.dobot.get_position()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))

        self.observation_space = spaces.Box(low=np.array([-np.pi / 2.0, 0, -0.174533, -np.pi / 2.0]),
                                            high=np.array([np.pi / 2.0, 1.48353, 1.65806, np.pi / 2.0]))


    def reward(self):
        tx, ty, tz = self.target_pos
        px, py, pz = self.dobot_ee_tip.get_position()
        r_dist_sq = (tx - px) ** 2 + (ty - py) ** 2 + (tz - pz) ** 2
        return -r_dist_sq

    def is_over(self):
        p_tip = np.array(self.dobot_ee_tip.get_position())
        dist_to_target = np.linalg.norm(p_tip - self.target_pos)

        return dist_to_target <= 0.05


    def _get_state(self):
        return np.array(self.dobot.get_joint_positions())


    def step(self, action):
        noise = np.random.normal(self.action_noise_mean, self.action_noise_variance, len(action))
        self.dobot.set_joint_target_velocities(action + noise)  # Execute action on arm, with actuator noise
        self.pr.step()  # Step the physics simulation
        return self._get_state(), self.reward(), self.is_over(), {}


    def reset(self):
        colliding = True
        while colliding:
            self.pr.set_configuration_tree(self.initial_dobot_state)
            self.dobot.set_control_loop_enabled(True)
            self.dobot.set_joint_target_velocities(np.zeros(4))
            self.dobot.set_joint_target_positions(self.observation_space.sample()) #, disable_dynamics=True)
            # Let things settle first...
            for i in range(100):
                self.pr.step()
            self.dobot.set_control_loop_enabled(False)
            colliding = self.dobot.check_arm_collision()


        return self._get_state()


    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.camera.capture_rgb()


    def close(self):
        self.pr.stop()
        self.pr.shutdown()
