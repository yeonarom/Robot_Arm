from gym.envs.registration import register

register(
    id='panda-v0',
    entry_point='envs.panda:PandaEnv',
)

register(
    id='ball-v0',
    entry_point='envs.ball:BallEnv',
)