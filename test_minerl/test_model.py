import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np

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
BATCH_SIZE = 32
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


n_observation_feats =  FRAME_STACK * 64 * 64 #  64 * 64 * 3 * FRAME_STACK 
print(f"num observation features: {n_observation_feats}")
n_actions = 14
done = False


# Defining the Q networks
policy_net = DQfD(n_observation_feats, n_actions).to(device)
policy_net = policy_net.float()
# target_net = DQfD(n_observations, n_actions).to(device)
# target_net.load_state_dict(policy_net.state_dict())

# optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
replay_memory = ReplayMemory(1000)


# Defining epsilon greedy action selection
def select_action(state):
    sample = random.random()
    if sample > EPS:
        print("Exploiting")
        with torch.no_grad():
            return torch.argmax(policy_net(state), dim=1) #policy_net(state).max(1)[1].view(1, 1)
    else:
        print("Exploring")
        return random.choice(list(action_names.keys()))
        # return actions[action_names[random.choice(list(action_names.keys))]]




# Optimizing the Q-network
def optimize_model(replay_memory, demo_replay_memory, BETA = 0.9):

    sample = random.random()
    if sample > BETA:
        print("Sampling from demo replay memory")
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = sample_demo_batch(demo_replay_memory, BATCH_SIZE, grayscale=True)
        batch_states = torch.reshape(torch.as_tensor(np.array(batch_states)), (BATCH_SIZE,-1)).float()
        batch_next_states = torch.reshape(torch.as_tensor(np.array(batch_next_states)), (BATCH_SIZE,-1)).float()
        batch_actions = np.array(batch_actions)
        batch_rewards = np.array(batch_rewards)
        batch_dones = np.array(batch_dones)

    else:
        print("Sampling from agent's replay memory")

    '''
    Optimize the Q-network either using the agent's self-explored replay memory or using demo data. 
    A variable defines the probability of sampling from either one 
    '''





    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()




for i in range(2):
    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = sample_demo_batch(demo_replay_memory, BATCH_SIZE, grayscale=True)
    
    batch_states = torch.reshape(torch.as_tensor(np.array(batch_states)), (BATCH_SIZE,-1)).float()
    print(batch_states.shape) # Shape must be [batch_size * frame_stack * 64 * 64] OR [batch_size * in_dims] 
    print(batch_actions)
    # action = select_action(batch_states)
    # print(action)
