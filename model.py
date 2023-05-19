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


# Setting up a device
# print(f"Is GPU available: {torch.cuda.is_available()}")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"


# Setting up a transition for the replay memory
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


# Defining the replay memory class for the agent's self-explored transitions
class ReplayMemory:
    '''
    Replay memory contains states and next_states that are stacked with FRAME_STACK number of observations. These are numpy arrays and not tensors.
    They are converted to tensors during optimization
    '''

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def append(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # samle returns a list of transitions with batch_size number of elements
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Defining the model class
class DQfD(nn.Module):

    def __init__(self, n_observation_feats, n_actions, batch_size):
        super(DQfD, self).__init__()
        self.layer1 = nn.Linear(n_observation_feats, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        self.batch_size = batch_size

    # Called with either one element to determine next action, or a batch
    def forward(self, x, for_optimization=True):
        # When using for optimization, a batch of inputs is passed in. In this case, reshape. When using for selecting actions, only one state is 
        # passed. In this case, the shape is already correctly set. Hence no reshaping is needed.
        if for_optimization: 
            x = torch.reshape(x, (self.batch_size,-1))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# Defining the dueling network architecture
class dueling_net(nn.Module):
    '''
    Dueling network architecture from "Dueling Network Architectures for Deep Reinforcement Learning". Conv layers modified to take shape of 64 x 64
    https://nn.labml.ai/rl/dqn/model.html
    '''
    def __init__(self, n_actions, frame_stack):
        super(dueling_net, self).__init__()

        # Conv output = [(W−K+2P)/S]+1 where W = input size, K = kernel size, P = padding, S = stride
        self.FRAME_STACK = frame_stack
        # self.batch_size = batch_size

        self.conv = nn.Sequential(
            # The first convolution layer takes a 64 x 64 frame and produces a 15 x 15 frame
            nn.Conv2d(in_channels=self.FRAME_STACK, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),

            # The second convolution layer takes a 15 x 15 and produces a 6 x 6 frame
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            nn.ReLU(),

            # The third convolution layer takes a 6 x 6 frame and produces a 5 x 5 frame
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1),
            nn.ReLU(),
        )

        # A fully connected layer takes the flattened
        self.lin = nn.Linear(in_features=5 * 5 * 64, out_features=512)
        self.activation = nn.ReLU()

        # This head gives the state value $V$
        self.state_value = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1),
        )

        # This head gives the action value $A$
        self.action_value = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=n_actions),
        )



    def forward(self, obs: torch.Tensor, for_optimization=True):
        # Convolution
        h = self.conv(obs)
        # Reshape for linear layers
        h = h.reshape((-1, 5 * 5 * 64))

        # Linear layer
        h = self.activation(self.lin(h))

        # $A$
        action_value = self.action_value(h)
        # $V$
        state_value = self.state_value(h)
        
        action_score_centered = action_value - action_value.mean(dim=-1, keepdim=True)
        q = state_value + action_score_centered

        # print("duelling net forward pass completed")
        return q



# Defining a modified DQfD loss function (1 step q learning loss + large margin classification loss + L2 regularization (implemented within adam))
class DQfD_Loss(nn.Module):
    def __init__(self):
        super(DQfD_Loss, self).__init__()


    def forward(self, policy_net, target_net, states, actions, rewards, next_states, dones, GAMMA, large_margin=True):
        # 1-step TD loss
        with torch.no_grad():
            next_state_max = torch.max(target_net(next_states), dim=1).values

        targets = rewards + GAMMA * next_state_max # torch.max(target_net(next_states), dim=1).values
        values = policy_net(states).gather(1,actions.view(-1,1)).view(-1,)

        if large_margin == True:
            # large margin loss
            lm1 = torch.max(policy_net(states), dim=1).values # Q value of the agent's actions as per its current policy
            lm2 = torch.eq(torch.argmax(policy_net(states), dim=1), actions).float() # Margin function: 0 if agent's action is the same as the expert's action 1 otherwise
            with torch.no_grad():
                lm3 = target_net(states).gather(1, actions.view(-1,1)).view(-1,) # Q value of the expert's action as per the target net (similar to why we use a frozen target)

            large_margin_loss = torch.mean(lm1+lm2+lm3)        

            return F.mse_loss(values, targets) + large_margin_loss
            
        else:
            return F.mse_loss(values, targets)



# Defining epsilon greedy action selection
def select_action(state, EPS, policy_net):
    state = state.to(device)
    sample = random.random()
    if sample > EPS:
        # print("Exploiting")
        with torch.no_grad():
            return torch.argmax(policy_net(state, for_optimization=False), dim=1).item()
    else:
        # print("Exploring")
        return action_names[random.choice(list(action_names.keys()))]
        # return action_list[action_names[random.choice(list(action_names.keys()))]]




# Defining the optimization for the Q-network
def optimize_model(optimizer, policy_net, target_net, replay_memory, demo_replay_memory, dqfd_loss, BATCH_SIZE = 32, BETA = 0.5, GAMMA=0.99):
    '''
    Optimize the Q-network either using the agent's self-explored replay memory or using demo data. 
    The variable BETA defines the probability of sampling from either one. This will later be replaced by some importance sampling factor
    '''

    sample = random.random()
    if sample > BETA:
        # print("Sampling from demo replay memory")
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = sample_demo_batch(demo_replay_memory, BATCH_SIZE, grayscale=True)
        batch_states, batch_next_states, batch_rewards, batch_actions = batch_states.to(device), batch_next_states.to(device), batch_rewards.to(device), batch_actions.to(device)
        loss = dqfd_loss(policy_net, target_net, batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, GAMMA, large_margin=True)

        # print(f"Loss: {loss}")


    else:
        # print("Sampling from agent's replay memory")
        batch_transitions = replay_memory.sample(BATCH_SIZE)

        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []

        for i in range(BATCH_SIZE):
            batch_states.append(batch_transitions[i].state)
            batch_next_states.append(batch_transitions[i].next_state)
            batch_rewards.append(batch_transitions[i].reward)
            batch_actions.append(batch_transitions[i].action)

        # batch_states = torch.reshape(torch.tensor(np.array(batch_states), dtype=torch.float32, requires_grad=True), (BATCH_SIZE,-1))
        batch_states = torch.tensor(np.array(batch_states), dtype=torch.float32, requires_grad=True)
        # batch_next_states = torch.reshape(torch.tensor(np.array(batch_next_states), dtype=torch.float32, requires_grad=True), (BATCH_SIZE,-1))
        batch_next_states = torch.tensor(np.array(batch_next_states), dtype=torch.float32, requires_grad=True)
        batch_actions = torch.tensor(np.array(batch_actions))
        batch_rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32, requires_grad=True)
        batch_dones = torch.tensor(np.array(batch_dones))

        # print(batch_states.shape)
        # print(batch_next_states.shape)
        # print(batch_actions.shape)
        # print(batch_rewards.shape)
        # print(batch_dones.shape)
        batch_states, batch_next_states, batch_rewards, batch_actions = batch_states.to(device), batch_next_states.to(device), batch_rewards.to(device), batch_actions.to(device)
        loss = dqfd_loss(policy_net, target_net, batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, GAMMA, large_margin=False)
        # print(f"Loss: {loss}")

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    # print("Optimizer steped ahead")
    return loss

