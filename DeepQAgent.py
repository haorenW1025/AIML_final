#  from agent_dir.agent import Agent
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

# pylint: disable=no-member

def nv_state_preprocessor(state):

    colors = ['G', 'R', 'Y']
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


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class DQN(nn.Module):
    def __init__(self, in_channels=3, action_num=5):

        super(DQN, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            #  Flatten(),
            #  nn.Linear(7*7*64, 512),
        )
        self.fc_value = nn.Linear(512, 1)
        self.fc_advantage = nn.Linear(512, action_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = self.main(x)
        value = self.fc_value(x)
        #  advantange = self.fc_advantage(x)
        #  q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        return value


class Agent_DQN():
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)
        self.device = self._prepare_gpu()
        self.discount_factor = 0.99
        self.initial_eps = 1
        self.final_eps = 0.025
        self.exploration_eps = 1000000
        self.batch_size = 32
        self.no_op_steps = 30
        self.replay_start = 5000
        self.memory = deque(maxlen=10000)
        self.episode = 30000
        self.current_Q = DQN().to(self.device)
        self.target_Q = DQN().to(self.device)
        self.target_Q.load_state_dict(self.current_Q.state_dict())
        self.optimizer = torch.optim.Adam(self.current_Q.parameters(), lr=0.00015, betas = (0.9, 0.999))
        #self.optimizer = torch.optim.RMSprop(self.current_Q.parameters(), lr=0.0001, alpha=0.95, eps=0.01)
        self.reward_list = []
        self.log_list = []
        self.save_dir = 'save_model'



        ##################
        # YOUR CODE HERE #
        ##################


    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        ##################
        # YOUR CODE HERE #
        ##################
        checkpoint = torch.load(self.args.check_path)
        self.current_Q.load_state_dict(checkpoint['state_dict'])
        self.current_Q.eval()
        self.current_Q.to(self.device)
        pass

    def epsilon(self, step):
        if step < self.exploration_eps:
            eps = self.final_eps + (self.initial_eps - self.final_eps) * (self.exploration_eps - step) / self.exploration_eps
            return eps
        else:
            return self.final_eps

    def train_replay(self):
        if len(self.memory) < self.replay_start:
            return
        mini_batch = random.sample(self.memory, self.batch_size)
        current_state, next_state, batch_action, batch_reward, batch_done = zip(*mini_batch)

        current_state = torch.stack(current_state).type(torch.FloatTensor).to(self.device).squeeze() / 255.0
        next_state = torch.stack(next_state).type(torch.FloatTensor).to(self.device).squeeze() / 255.0
        batch_action = torch.stack(batch_action).type(torch.LongTensor).to(self.device).squeeze()
        batch_reward = torch.stack(batch_reward).type(torch.FloatTensor).to(self.device).squeeze()
        batch_done = torch.stack(batch_done).type(torch.FloatTensor).to(self.device).squeeze()

        q_prediction = self.current_Q(current_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        target_q = ((self.target_Q(next_state).detach().max(-1)[0] * (1 - batch_done) * self.discount_factor) + batch_reward)

        self.optimizer.zero_grad()
        loss = F.smooth_l1_loss(q_prediction, target_q)
        loss = loss.clamp(-1, 1)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################

        step = 0
        loss = []
        current_eps = 1

        for e in range(1, self.episode+1):
            observation = self.env.reset()
            #observation = observation.astype(np.float64)
            done = False
            reward_sum = 0

            while not done:
                #self.env.env.render()
                if step > self.replay_start:
                    current_eps = self.epsilon(step)
                action = random.choice([0,1,2,3,4]) if random.random() < current_eps \
                    else self.make_action(observation, False)
                next_ob, reward, done, info = self.env.step(action + 1)
                #next_ob = next_ob.astype(np.float64)
                reward_sum += reward
                step += 1
                self.memory.append((
                    torch.ByteTensor([observation]),
                    torch.ByteTensor([next_ob]),
                    torch.ByteTensor([action]),
                    torch.ByteTensor([reward]),
                    torch.ByteTensor([done])
                ))
                observation = next_ob

                if step % 4 == 0:
                    loss.append(self.train_replay())
                if step % 1000 == 0:
                    self.target_Q.load_state_dict(self.current_Q.state_dict())

            self.reward_list.append(reward_sum)

            self.logger.info("Episode: {} | Step: {} | Reward: {} | Loss: {}".format(e, step, reward_sum, loss[-1]))

            if e % 10 == 0:
                log = {
                    'episode': e,
                    'loss': loss,
                    'step': step,
                    'latest_reward': self.reward_list
                }
                self.log_list.append(log)
                checkpoint = {
                    'log': self.log_list,
                    'state_dict': self.current_Q.state_dict()
                }
                torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoint_episode{}.pth.tar'.format(e)))
                self.logger.info('save checkpoint_episode{}.pth.tar!'.format(e))
        pass

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        observation = np.array(observation).astype(np.float32) / 255.0
        action = self.current_Q((torch.FloatTensor(observation).unsqueeze(0)).to(self.device)).max(-1)[1].data[0]
        return action.item() + 1 if test else action.item()

    def _prepare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        return device

if __name__ == "__main__":
    from torchsummary import summary
    model = DQN();
    summary(model, input_size=(3, 200, 200), device='cpu')
