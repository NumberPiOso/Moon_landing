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
goal_steps = 200
score_requirement = 50
initial_games = 1000

def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(200):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print( action)
            if done:
                break


some_random_games_first()