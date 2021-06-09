# Robotic Experiment Setup for Collaboration on Robust GAIL project

This repo provides a simulated environment for replicating an RL/Learning-from-Demonstration reaching task on a [Franka Emika Panda](https://www.franka.de/) 7-DOF robot arm.

The "__expert__" system runs in a perfect, noise-free environment, and reaches towards a target sphere using velocites provided by an off-the-shelf IK-solver available in [CoppeliaSim](https://www.coppeliarobotics.com/).

The "__learned_agent__" provides desired velocities for each of the 7 arm-joints at each time step. However, the agent may not run in the same "perfect" environment as the expert, as noise is added to the agent's actions such that the desired and realised joint-velocities may differ.

![Robot Reaching](pics/robotReach.png)

## Setup
---

1. Download and install CoppeliaSim __VERSION 4.1__ : [(Link for Ubuntu 20)](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)
2. Clone and install [PyRep](https://github.com/stepjam/PyRep). Setup instructions are given [at the repo](https://github.com/stepjam/PyRep), but here's a summary:

	- Clone the repo

			git clone https://github.com/stepjam/PyRep.git
			cd PyRep

	- Add the following to your ~/.bashrc file: (NOTE: the 'EDIT ME' in the first line)

			export COPPELIASIM_ROOT=EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR
			export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
			export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

	- Install Dependencies and PyRep itself

			pip3 install -r requirements.txt
			pip3 install .


## Usage
---
Run the following command:

		python rl_env.py [-h] [--headless] agent_type episode_length num_episodes

Where the parameters are:
- `episode-length` - the maximum length of an episode (e.g, 200)
- `num-episodes` - is the total number of episodes to run for (e.g., 20)
- `--headless` - flag determines whether the simulations are run headless (in the command line without GUI visuals).
- `agent-type` -  Can be either `agent` (run using the RL policy defined in the `Agent` class, with noisy actions) or `expert` (run with an IK solver and perfect noisless actions).

## Implementing your own RL Agent
---

### Modifying the Agent

The file `rl_env.py` contains an `Agent()` class. It has the following methods:

- `act(self, state) -> np.ndarray` - Takes in the current state (A (7 + 7 + 3)-dimensional vector consisting of the 7 arm-joint positions, 7 arm-joint velocities, and XYZ coordinates of the target object). Outputs a 7-dimensional action (The desired velocities for each joint)
- `learn(self, replay_buffer)` - Takes in a total history (so far) of `(state, action, reward, next_state)` tuples at each time step, and uses them to update the agent's policy

To implement your own agent, sub-class the `Agent` class, and modify the above methods to with your own RL-algorithm.

### Modifying the Environment

`rl_env.py` also contains a `NoisyReacherEnv` class. To modify the environment, the following parts are relevant:

- The `reward(self)` function - Outputs the reward the agent receives based on the current state of the world. The default is to provide a reward equal to the negative squared distance from the robot end-effector to the target-objectg position.

- The `action_noise_mean` and `action_noise_var` variables define the mean/variance of the random noise distribution which is added to the agent's desired joint velocities before translating them to actual controlled velocities. By default, the expert environment has no noise, and the agent environment has noise with mean 0.1 and variance 0.05

Within `rl_env.py`, there is a class called `Agent()` which has an `act` method where you can put in your own policy. You can also change the state space / reward function by looking at the `NoisyReacherEnv` class.
