import gym
import random
import numpy as np
import tensorflow
# import tflearn
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.estimator import regression
from collections import Counter

LR = 1e-3
env = gym.make('LunarLander-v2')
env.reset()
for _ in range(100):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
