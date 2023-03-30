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



# Defining the replay memory class for the agent's self-explored transitions
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


# Defining a modified DQfD loss function (1 step q learning loss + large margin classification loss + L2 regularization (implemented within adam))
class DQfD_Loss(nn.Module):
    def __init__(self):
        super(DQfD_Loss, self).__init__()


    def forward(self, policy_net, target_net, states, actions, rewards, next_states, dones, GAMMA):

        # terminal_mask = torch.as_tensor(np.array(dones)).ge(1)
        # non_terminal_mask = ~terminal_mask


        # terminal_states = torch.masked_select(states.T, terminal_mask) # Not to be confused with terminal next_states. Terminal next_states are the actual terminal states. 
        # terminating_actions = torch.masked_select(actions , terminal_mask)
        # terminal_next_states = torch.masked_select(next_states.T, terminal_mask)
        # terminal_rewards = torch.masked_select(rewards, terminal_mask)

        # non_terminal_states = torch.masked_select(states.T, non_terminal_mask) # Not to be confused with terminal next_states. Terminal next_states are the actual terminal states. 
        # non_terminating_actions = torch.masked_select(actions , non_terminal_mask)
        # non_terminal_next_states = torch.masked_select(next_states.T, non_terminal_mask)
        # non_terminal_rewards = torch.masked_select(rewards, non_terminal_mask)

        # 1-step TD loss
        targets = rewards + GAMMA * torch.max(target_net(next_states), dim=1).values
        values = policy_net(states).gather(1,actions.view(-1,1)).view(-1,)

        # large margin loss

        lm1 = torch.max(policy_net(batch_states), dim=1).values # Q value of the agent's actions as per its current policy
        lm2 = torch.eq(torch.argmax(policy_net(batch_states), dim=1), batch_actions).float() # Margin function: 0 if agent's action is the same as the expert's action 1 otherwise
        lm3 = target_net(batch_states).gather(1, batch_actions.view(-1,1)).view(-1,) # Q value of the expert's action as per the target net (similar to why we use a frozen target)

        large_margin_loss = torch.mean(lm1+lm2+lm3)        

        return F.mse_loss(values, targets) + large_margin_loss




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
n_actions = 15


# Defining the Q networks
policy_net = DQfD(n_observation_feats, n_actions).to(device)
policy_net = policy_net.float()
target_net = DQfD(n_observation_feats, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR, weight_decay=0.01) # Weight decay is L2 regularization
replay_memory = ReplayMemory(1000)
dqfd_loss = DQfD_Loss()


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
def optimize_model(replay_memory, demo_replay_memory, BETA = 0, GAMMA=GAMMA):
    '''
    Optimize the Q-network either using the agent's self-explored replay memory or using demo data. 
    The variable BETA defines the probability of sampling from either one. This will later be replaced by some importance sampling factor
    '''

    sample = random.random()
    if sample > BETA:
        print("Sampling from demo replay memory")
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = sample_demo_batch(demo_replay_memory, BATCH_SIZE, grayscale=True)


    else:
        print("Sampling from agent's replay memory")



    # Compute loss
    with torch.no_grad():
        loss = dqfd_loss(policy_net, target_net, batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, GAMMA)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()




for i in range(1):
    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = sample_demo_batch(demo_replay_memory, BATCH_SIZE, grayscale=True)
    
    q_values = policy_net(batch_states)

    print("BATCH SHAPES")
    print(f"States shape {batch_states.shape}")
    print(f"Actions shape {batch_actions.shape}")
    print(f"Next states shape {batch_next_states.shape}")
    print(f"Rewards shape {batch_rewards.shape}")
    print(f"Dones shape {batch_dones.shape}")
    print(f"Q Values shape: {q_values.shape}")

