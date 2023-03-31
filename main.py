import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

import gym
import minerl
from minerl.data import BufferedBatchIter

import model
from demo_sampling import sample_demo_batch



# Setting up a device
print(f"Is GPU available: {torch.cuda.is_available()}")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


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
replay_memory = model.ReplayMemory(1000)

n_observation_feats =  FRAME_STACK * 64 * 64 #  64 * 64 * 3 * FRAME_STACK 
print(f"num observation features: {n_observation_feats}")
n_actions = 15


# Defining the Q networks
policy_net = model.DQfD(n_observation_feats, n_actions).to(device)
policy_net = policy_net.float()
target_net = model.DQfD(n_observation_feats, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

# Defining the loss function and optimizer
optimizer = optim.Adam(policy_net.parameters(), lr=LR, weight_decay=0.01) # Weight decay is L2 regularization
dqfd_loss = model.DQfD_Loss()




for i in range(1):
    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = sample_demo_batch(demo_replay_memory, BATCH_SIZE, grayscale=True)
    
    # q_values = policy_net(batch_states)

    print("BATCH SHAPES")
    print(f"States shape {batch_states.shape}")
    print(f"Actions shape {batch_actions.shape}")
    print(f"Next states shape {batch_next_states.shape}")
    print(f"Rewards shape {batch_rewards.shape}")
    print(f"Dones shape {batch_dones.shape}")
    # print(f"Q Values shape: {q_values.shape}")

    model.optimize_model(optimizer, policy_net, target_net, replay_memory, demo_replay_memory, dqfd_loss, BATCH_SIZE = BATCH_SIZE, BETA = 0, GAMMA=GAMMA)