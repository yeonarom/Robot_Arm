import gym

import sys
sys.path.append('/home/yeonarom/robot/ros2/panda_gym_ws/panda-rl/environments/simple_env/envs')
from panda import PandaEnv
from triple_ball import TripleBallEnv
from single_ball import SingleBallEnv

from stable_baselines3 import PPO

model_path = '/home/yeonarom/robot/ros2/panda_gym_ws/panda-rl/agents/models/single_ball-v1'

# env = gym.make("PandaReach-v2")
env = SingleBallEnv()

model = PPO.load(model_path)

observation = env.reset()

for _ in range(1000):
    action, _states = model.predict(observation)
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()

env.close()
