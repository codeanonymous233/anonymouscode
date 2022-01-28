'''
Policy networks for collecting trajectories in a dynamics model's learning.
'''

import torch
import torch.nn as nn
from torch.autograd import Variable


class RandomPolicy(nn.Module):

    def __init__(self, env, discrete_action_space=False):
        super().__init__()
        self.env = env
        self.discrete_action_space=discrete_action_space

    def forward(self, state):
        if self.discrete_action_space:
            action=Variable(torch.tensor(self.env.action_space.sample())) 
        else:
            action=Variable(torch.FloatTensor(self.env.action_space.sample())) 
            
        return action

    

