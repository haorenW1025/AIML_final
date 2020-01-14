import numpy as np
from ple import PLE

from ple.games.waterworld import WaterWorld
import time
import os


class NaiveAgent():
    """
            This is our naive agent. It picks actions at random!
    """

    def __init__(self, actions):
        self.actions = actions

    def pickAction(self, reward, obs):
        return self.actions[np.random.randint(len(self.actions))]


# preprocessor example
def nv_state_preprocessor(state):

    colors = ['G', 'R', 'Y']
    val = state.values()
    step = state['step']
    x_pos = round(state['player_x'])
    y_pos = round(state['player_y'])
    x_vel = state['player_velocity_x']
    y_vel = state['player_velocity_y']
    tmp = state['creep_pos']
    all_pos = np.array([])
    for color in colors:
        for position in tmp[color]:
            k = np.array(position)
            dis_x = x_pos - k[0]
            dis_y = y_pos - k[1]
            all_pos = np.append(all_pos, (round(dis_x), round(dis_y)))
    all_dist = np.array([])
    tmp = state['creep_dist']

    for color in colors:
        k = np.array(tmp[color])/362.1
        all_dist = np.append(all_dist, k)

    print step
    all_state = np.append([step, x_pos, y_pos, x_vel, y_vel], all_pos)

    return all_state.flatten()


###################################


# Don't display window.
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

# create our game
force_fps = True  # slower speed
reward = 0.0
game = WaterWorld()

# make a PLE instance.
p = PLE(game, force_fps=force_fps, display_screen=True,
        state_preprocessor=nv_state_preprocessor)

# our Naive agent!
agent = NaiveAgent(p.getActionSet())
print p.getActionSet()

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
        print state
        action = agent.pickAction(reward, obs)
        reward = p.act(action)  # reward after an action
    score = game.getScore()
    print "Trial no.{:02d} : score {:0.3f}".format(i,score)

# Screen Shot
#     p.saveScreen("screen_capture.png")
