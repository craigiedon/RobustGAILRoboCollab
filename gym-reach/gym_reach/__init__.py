from gym.envs.registration import register

register(
    id='reachNoisy-v0',
    entry_point='gym_reach.envs:ReachEnv'
)

register(
    id='reachNoisyFixed-v0',
    entry_point='gym_reach.envs:ReachEnvFixed'
)

register(
    id='reachDobotFixed-v0',
    entry_point='gym_reach.envs:ReachDobotFixed'
)

register(
    id='reachDobotMulti-v0',
    entry_point='gym_reach.envs:ReachDobotMulti'
)
