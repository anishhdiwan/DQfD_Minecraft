import torch
import torch.optim as optim
import cv2
import numpy as np

import gym
import minerl
from minerl.data import BufferedBatchIter

import model # Import the classes and functions defined in model.py
from utils import stack_observations, pad_state
from actions import actions as action_list
# from demo_sampling import sample_demo_batch



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
BATCH_SIZE = 2
GAMMA = 0.99
EPS = 0.5
TAU = 0.005
LR = 1e-4



# Creating the environment (this may take a few minutes) and setting up the data sampling iterator
env = gym.make('MineRLTreechop-v0')
print("Gym.make done")

# Initializing the generator
# Download the dataset before running this script
data = minerl.data.make('MineRLTreechop-v0')
iterator = BufferedBatchIter(data)
demo_replay_memory = iterator.buffered_batch_iter(batch_size=FRAME_STACK, num_epochs=1) # The batch_size here refers to the number of consequtive frames
replay_memory = model.ReplayMemory(1000)

n_observation_feats =  FRAME_STACK * 64 * 64 #  64 * 64 * 3 * FRAME_STACK 
n_actions = 15
print(f"num observation features: {n_observation_feats}")
print(f"num actions: {n_actions}")



# Defining the Q networks
policy_net = model.DQfD(n_observation_feats, n_actions).to(device)
policy_net = policy_net.float()
target_net = model.DQfD(n_observation_feats, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

# Defining the loss function and optimizer
optimizer = optim.Adam(policy_net.parameters(), lr=LR, weight_decay=0.0001) # Weight decay is L2 regularization
dqfd_loss = model.DQfD_Loss()




# for i in range(1):

#     # model.optimize_model(optimizer, policy_net, target_net, replay_memory, demo_replay_memory, dqfd_loss, BATCH_SIZE=BATCH_SIZE, BETA = 0, GAMMA=GAMMA)
#     batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = sample_demo_batch(demo_replay_memory, BATCH_SIZE, grayscale=True)
#     print(batch_states.shape)



# Main function
num_episodes = 1
num_steps = 10

for i_episode in range(num_episodes):

    # Initialize the environment and get it's state
    obs = env.reset()
    print("Reset Successful")
    obs_gray = cv2.cvtColor(obs['pov'], cv2.COLOR_BGR2GRAY)
    # state = torch.tensor(stack_observations(obs_gray, FRAME_STACK), dtype=torch.float32) # Stacking observations together to form a state
    state = stack_observations(obs_gray, FRAME_STACK)
    print(f"First state shape: {state.shape}")

    for t in range(num_steps):
        # action = model.select_action(torch.reshape(state, (1,-1)), EPS, policy_net)
        action = model.select_action(torch.reshape(torch.tensor(state, dtype=torch.float32), (1,-1)), EPS, policy_net)
        print(f"action: {action}")
        next_state = np.zeros(state.shape)
        reward = 0
        done = False
        for i in range(FRAME_STACK):
            if not done:
                next_observation, next_reward, done, _ = env.step(action_list[action])
                next_obs_gray = cv2.cvtColor(next_observation['pov'], cv2.COLOR_BGR2GRAY)
                next_state[i] = next_obs_gray
                reward += next_reward

        print(f"Completed {FRAME_STACK} transitions")

        # Store the transition in memory
        if not done:
            # next_state = torch.tensor(next_state, dtype=torch.float32, requires_grad=True)
            replay_memory.append(state, action, reward, next_state)
        else:
            # next_state = torch.tensor(pad_state(next_state, FRAME_STACK), dtype=torch.float32, requires_grad=True)
            next_state = pad_state(next_state, FRAME_STACK)
            replay_memory.append(state, action, reward, next_state)

        # Move to the next state
        state = next_state

        # Sampling from the demo replay until the replay memory has at least BATCH_SIZE number of transitions
        if len(replay_memory) < BATCH_SIZE:
            BETA = 0.5
        else:
            BETA = 0.5

        # Perform one step of the optimization (on the policy network)
        model.optimize_model(optimizer, policy_net, target_net, replay_memory, demo_replay_memory, dqfd_loss, BATCH_SIZE=BATCH_SIZE, BETA = 0.5, GAMMA=GAMMA)
        print("Completed one step of optimization")

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        print("Completed one step of soft update")

        if done:
            # episode_durations.append(t + 1)
            break

        print("--------------")



# print('Complete')
# plot_durations(show_result=True)
# plt.ioff()
# plt.show()
# print(f"Sampling replay_memory: {replay_memory.sample(1)[0].state.shape}")
# print(f"Sampling replay_memory: {replay_memory.sample(1)[0].next_state.shape}")