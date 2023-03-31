import random
from collections import namedtuple, deque
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Actions are defined as dictionaries in mineRL. A smaller list of actions is defined separately and these are imported here for simplicity
from actions import action_names
from demo_sampling import sample_demo_batch


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
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)



# Defining a modified DQfD loss function (1 step q learning loss + large margin classification loss + L2 regularization (implemented within adam))
class DQfD_Loss(nn.Module):
    def __init__(self):
        super(DQfD_Loss, self).__init__()


    def forward(self, policy_net, target_net, states, actions, rewards, next_states, dones, GAMMA, large_margin=True):
        # 1-step TD loss
        targets = rewards + GAMMA * torch.max(target_net(next_states), dim=1).values
        values = policy_net(states).gather(1,actions.view(-1,1)).view(-1,)

        if large_margin == True:
            # large margin loss
            lm1 = torch.max(policy_net(states), dim=1).values # Q value of the agent's actions as per its current policy
            lm2 = torch.eq(torch.argmax(policy_net(states), dim=1), actions).float() # Margin function: 0 if agent's action is the same as the expert's action 1 otherwise
            lm3 = target_net(states).gather(1, actions.view(-1,1)).view(-1,) # Q value of the expert's action as per the target net (similar to why we use a frozen target)

            large_margin_loss = torch.mean(lm1+lm2+lm3)        

            return F.mse_loss(values, targets) + large_margin_loss
            
        else:
            return F.mse_loss(values, targets)



# Defining epsilon greedy action selection
def select_action(state, EPS, policy_net):
    sample = random.random()
    if sample > EPS:
        print("Exploiting")
        with torch.no_grad():
            return torch.argmax(policy_net(state), dim=1).item() #policy_net(state).max(1)[1].view(1, 1)
    else:
        print("Exploring")
        return action_names[random.choice(list(action_names.keys()))]
        # return action_list[action_names[random.choice(list(action_names.keys()))]]




# Defining the optimization for the Q-network
def optimize_model(optimizer, policy_net, target_net, replay_memory, demo_replay_memory, dqfd_loss, BATCH_SIZE = 32, BETA = 0, GAMMA=0.99):
    '''
    Optimize the Q-network either using the agent's self-explored replay memory or using demo data. 
    The variable BETA defines the probability of sampling from either one. This will later be replaced by some importance sampling factor
    '''

    sample = random.random()
    if sample > BETA:
        print("Sampling from demo replay memory")
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = sample_demo_batch(demo_replay_memory, BATCH_SIZE, grayscale=True)
        loss = dqfd_loss(policy_net, target_net, batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, GAMMA, large_margin=True)
        print(f"Loss: {loss}")


    else:
        print("Sampling from agent's replay memory")
        # code to be added
        loss = dqfd_loss(policy_net, target_net, batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, GAMMA, large_margin=False)
        print(f"Loss: {loss}")


    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    # print("Optimizer steped ahead")

