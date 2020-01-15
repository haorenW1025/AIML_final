#  from agent_dir.agent import Agent
from __future__ import print_function
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import torch.nn as nn
import random
import os
import sys
import time

from ple import PLE
from ple.games.waterworld import WaterWorld

os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

class SimpleAgent():

    def __init__(self, actions):
        self.actions = actions

    def pickAction(self, state):
        position_x = state[2]
        position_y = state[3]
        if position_x >= 0 and position_y >= 0:
            return self.actions[random.choice([0, 3])]
        elif position_x >= 0 and position_y < 0:
            return self.actions[random.choice([1, 3])]
        elif position_x < 0 and position_y >= 0:
            return self.actions[random.choice([0, 2])]
        elif position_x < 0 and position_y < 0:
            return self.actions[random.choice([1, 2])]


def nv_state_preprocessor(state):

    velocity_feature_x = 0
    velocity_feature_y = 0
    position_feature_x = 0
    position_feature_y = 0
    colors = ['G', 'R', 'Y']
    x_pos = round(state['player_x'])
    y_pos = round(state['player_y'])
    x_vel = state['player_velocity_x']
    y_vel = state['player_velocity_y']
    if x_vel > 0:
        velocity_feature_x = 1
    elif x_vel < 0:
        velocity_feature_x = -1

    if y_vel > 0:
        velocity_feature_y = 1
    elif y_vel < 0:
        velocity_feature_y = -1
    tmp = state['creep_pos']
    creep_x_pos = np.array([])
    creep_y_pos = np.array([])
    for color in colors:
        for position in tmp[color]:
            k = np.array(position)
            dis_x = k[0] - x_pos
            dis_y = k[1] - y_pos
            creep_x_pos = np.append(creep_x_pos, dis_x)
            creep_y_pos = np.append(creep_y_pos, dis_y)
    all_dist = np.array([])
    tmp = state['creep_dist']

    for color in colors:
        k = np.array(tmp[color])/362.1
        all_dist = np.append(all_dist, k)

    shortest_index = -1
    shortest_dis = 1000000
    shortest_dis = all_dist.min()
    shortest_index = np.where(all_dist == shortest_dis)
    #  print(shortest_dis)
    #  print(shortest_index)


    shortest_x = creep_x_pos[shortest_index]
    shortest_y = creep_y_pos[shortest_index]
    if shortest_x > 0:
        position_feature_x = 1
    elif shortest_x < 0:
        position_feature_x = -1

    if shortest_y > 0:
        position_feature_y = 1
    elif shortest_y < 0:
        position_feature_y = -1

    all_feature = np.array([])
    all_feature = np.append(all_feature, (velocity_feature_x, velocity_feature_y, position_feature_x, position_feature_y))

    return all_feature.flatten()

force_fps = True  # slower speed
reward = 0.0
game = WaterWorld()

# make a PLE instance.
p = PLE(game, force_fps=force_fps, display_screen=True,
        state_preprocessor=nv_state_preprocessor)

# our Naive agent!
agent = SimpleAgent(p.getActionSet())
print(p.getActionSet())

# init agent and game.
p.init()


# start our loop
score = 0.0
for i in range(10):
    # if the game is over
    if p.game_over():
        p.reset_game()
    while p.game_over() == False:
        obs = p.getScreenRGB()
        state = p.getGameState()
        action = agent.pickAction(state)
        reward = p.act(action)  # reward after an action
    score = game.getScore()
    print("Trial no.{:02d} : score {:0.3f}".format(i,score))

