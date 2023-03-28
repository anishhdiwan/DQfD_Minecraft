import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
import minerl
from minerl.data import BufferedBatchIter

# Actions are defined as dictionaries in mineRL. A smaller list of actions is defined separately and these are imported here for simplicity
import sys
sys.path.append('../')
from actions import actions, action_names
from demo_sampling import sample_demo_batch

# Setting up a device
print(f"Is GPU available: {torch.cuda.is_available()}")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Setting up a transition for the replay memory
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def append(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Defining the model class
class DQfD(nn.Module):

    def __init__(self, n_observation_feats, n_actions):
        super(DQfD, self).__init__()
        self.layer1 = nn.Linear(n_observation_feats, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# Defining model hyper-parameters
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS is the epsilon greedy exploration probability
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
FRAME_STACK = 2
BATCH_SIZE = 64
GAMMA = 0.99
EPS = 0
TAU = 0.005
LR = 1e-4


# Creating the environment (this may take a few minutes) and setting up the data sampling iterator
# env = gym.make('MineRLTreechop-v0')
# print("Gym.make done")

# Initializing the generator
# Download the dataset before running this script
data = minerl.data.make('MineRLTreechop-v0')
iterator = BufferedBatchIter(data)
demo_replay_memory = iterator.buffered_batch_iter(batch_size=FRAME_STACK, num_epochs=1) # The batch_size here refers to the number of consequtive frames


n_observation_feats = 64 * 64 * 3 * FRAME_STACK 
print(f"num observation features: {n_observation_feats}")
n_actions = 14
done = False


# Defining the Q networks
# policy_net = DQfD(n_observation_feats, n_actions).to(device)
# target_net = DQfD(n_observations, n_actions).to(device)
# target_net.load_state_dict(policy_net.state_dict())

# optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(1000)


# Defining epsilon greedy action selection
def select_action(state):
    sample = random.random()
    if sample > EPS:
        print("Exploiting")
        with torch.no_grad():
            return policy_net(state) #policy_net(state).max(1)[1].view(1, 1)
    else:
        print("Exploring")
        return random.choice(list(action_names.keys()))
        # return actions[action_names[random.choice(list(action_names.keys))]]



for i in range(2):
    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = sample_demo_batch(demo_replay_memory, BATCH_SIZE)
    

    # batch_states = torch.as_tensor(batch_states)
    # print(batch_states.shape)
    # print(select_action(torch.reshape(batch_states, (-1,128))))


