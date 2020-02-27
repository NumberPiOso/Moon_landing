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
score_requirement = 0
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

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0, 4)
            observation, reward, done, info = env.step(action)
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            
            prev_observation = observation
            score += reward
            if done:
                break

        if score > score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                output = [0,0,0,0]
                output[data[1]] = 1
                training_data.append([data[0], output])

        env.reset()
        scores.append(score)
    training_data_save = np.array(data)
    np.save('saved.npy', training_data_save)

    print('Average accepted score', np.mean(accepted_scores))
    print('Median accepted score', np.median(accepted_scores))
    print(accepted_scores)

    print('Mean all scores ', np.mean(scores))
    print('Median all score', np.median(scores))


# some_random_games_first()
initial_population()