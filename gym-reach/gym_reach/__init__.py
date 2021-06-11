from gym.envs.registration import register

register(
    id='reachAbstract-v0',
    entry_point='gym_reach.envs:ReachEnv'
)

register(
    id='reachPerfectExp-v0',
    entry_point='gym_reach.envs:ReachEnvPerfectExpert'
)

register(
    id='reachNoisy-v0',
    entry_point='gym_reach.envs:ReachEnvNoisy'
)