# Robotic Experiment Setup for Collaboration on Robust GAIL project

## Setup

1. Download and install [CoppeliaSim](https://www.coppeliarobotics.com/)
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

Run the following command:

		python rl_env.py [-h] [--headless] agent_type episode_length num_episodes

Where `episode-length` is the maximum length of an episode (e.g, 200), `num-episodes` is the total number of episodes to run for (e.g., 20), and the `--headless` flag determines whether the simulations are run headless in the command line without GUI visuals. Agent type can be either `agent` (in which case the program will run using the RL policy defined in the `Agent` class, with noisy actions) or `expert` (in which case the program will run with an IK solver and perfect noisless actions).

## Implementing your own RL Agent

Within `rl_env.py`, there is a class called `Agent()` which has an `act` method where you can put in your own policy. You can also change the state space / reward function by looking at the `NoisyReacherEnv` class.