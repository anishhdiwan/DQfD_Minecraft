import torch
import torch.optim as optim
import cv2
import numpy as np
from tqdm import tqdm
import psutil
import gym
import minerl
from minerl.data import BufferedBatchIter

import model # Import the classes and functions defined in model.py
from utils import stack_observations, pad_state
from actions import actions as action_list
from buffered_batch_iter_patches import optionally_fill_buffer_patch, buffered_batch_iter_patch
# from demo_sampling import sample_demo_batch

BufferedBatchIter.optionally_fill_buffer = optionally_fill_buffer_patch
BufferedBatchIter.buffered_batch_iter = buffered_batch_iter_patch

from torch.utils.tensorboard import SummaryWriter


# Setting up a device
print(f"Is GPU available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


# Defining model hyper-parameters
# BATCH_SIZE is the number of transitions sampled from the replay memory. A batch of inputs is sampled and fed through the optimizer when training the policy network
# GAMMA is the discount factor
# EPS is the epsilon greedy exploration probability
# TAU is the update rate of the target network
# LR is the learning rate of the optimizer
FRAME_STACK = 4
BATCH_SIZE = 32
GAMMA = 0.95
EPS = 0.1
TAU = 0.005
LR = 1e-4
num_episodes = 8
num_steps = 1500
save_checkpoint = 500 # save the model after these many steps
pre_train_steps = int(5*num_steps)
RUN_NAME = "HP_combo_1"
logdir = f"runs/frame_stack:{FRAME_STACK}_|batch_size:{BATCH_SIZE}_|gamma:{GAMMA}_|eps:{EPS}_|tau:{TAU}_|lr:{LR}_|episodes:{num_episodes}_|steps:{num_steps}_|run:{RUN_NAME}"
save_path = f"saved_models/frame_stack:{FRAME_STACK}_|batch_size:{BATCH_SIZE}_|gamma:{GAMMA}_|eps:{EPS}_|tau:{TAU}_|lr:{LR}_|episodes:{num_episodes}_|steps:{num_steps}_|run:{RUN_NAME}.pt"

# Setting up the tensorboard summary writer
writer = SummaryWriter(log_dir=logdir)


# Creating the environment (this may take a few minutes) and setting up the data sampling iterator
env = gym.make('MineRLTreechop-v0')
print("Gym.make done")

# Enable logging in minerl 
# import logging
# logging.basicConfig(level=logging.DEBUG)

# Initializing the generator
# Download the dataset before running this script
data = minerl.data.make('MineRLTreechop-v0')
iterator = BufferedBatchIter(data, buffer_target_size=5000)
demo_replay_memory = iterator.buffered_batch_iter(batch_size=FRAME_STACK) # The batch_size here refers to the number of consequtive frames

replay_memory = model.ReplayMemory(5000)
print("Replay memory & demo replay memory initialized")


n_actions = 15
# print(f"num actions: {n_actions}")


# Choosing a deep architecture:
# architecture = "simple"  
architecture = "duelling_net"

if architecture == "simple":
    # Defining the simple model Q networks
    policy_net = model.DQfD(n_observation_feats, n_actions, BATCH_SIZE).to(device)
    policy_net = policy_net.float()
    target_net = model.DQfD(n_observation_feats, n_actions, BATCH_SIZE).to(device)
    target_net.load_state_dict(policy_net.state_dict())

elif architecture == "duelling_net":
    # Defining the duelling network Q networks
    policy_net = model.dueling_net(n_actions, FRAME_STACK).to(device)
    policy_net = policy_net.float()
    target_net = model.dueling_net(n_actions, FRAME_STACK).to(device)
    target_net.load_state_dict(policy_net.state_dict())


# Defining the loss function and optimizer
optimizer = optim.Adam(policy_net.parameters(), lr=LR, weight_decay=1e-5) # Weight decay is L2 regularization
dqfd_loss = model.DQfD_Loss()





# Metrics
'''
- Loss vs num steps
- Episode return vs episode number
- num steps in episode vs episodes 
'''
total_steps = 0

# Main function
for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    obs = env.reset()
    print("Reset Successful")
    obs_gray = cv2.cvtColor(obs['pov'], cv2.COLOR_BGR2GRAY)
    # Stacking observations together to form a state
    state = stack_observations(obs_gray, FRAME_STACK)
    # print(f"First state shape: {state.shape}")
    
    # Metrics
    episode_return = 0
    episode_steps = 0
    loop = tqdm(range(num_steps))
    for t in loop:
        loop.set_description(f"Episode {i_episode} Steps | CPU {psutil.cpu_percent()} | RAM {psutil.virtual_memory().percent}")
        if architecture == "simple":
            action = model.select_action(torch.reshape(torch.tensor(state, dtype=torch.float32), (1,-1)), EPS, policy_net)
        elif architecture == "duelling_net":
            temp = torch.tensor(state, dtype=torch.float32)
            shape = list(temp.shape)
            shape.insert(0,1)
            action = model.select_action(temp.view(tuple(shape)), EPS, policy_net)
            # # Adding the model's graph in tensorboard
            # writer.add_graph(policy_net, temp.view(tuple(shape)))
            # writer.close()
        
        # print(f"action: {action}")
        next_state = np.zeros(state.shape)
        reward = 0
        done = False
        for i in range(FRAME_STACK):
            if not done:
                next_observation, next_reward, done, _ = env.step(action_list[action])
                next_obs_gray = cv2.cvtColor(next_observation['pov'], cv2.COLOR_BGR2GRAY)
                next_state[i] = next_obs_gray
                reward += next_reward

        # print(f"Completed {FRAME_STACK} transitions")

        # Store the transition in the agent's self-sampled memory
        if not done:
            replay_memory.append(state, action, reward, next_state)
        else:
            next_state = pad_state(next_state, FRAME_STACK)
            replay_memory.append(state, action, reward, next_state)

        # Move to the next state
        state = next_state

        # Sampling from the demo replay until the replay memory has at least BATCH_SIZE number of transitions
        # if len(replay_memory) < BATCH_SIZE:
        if (total_steps < pre_train_steps) or (len(replay_memory) < BATCH_SIZE):
            BETA = 0
        else:
            BETA = (total_steps - pre_train_steps)/(num_steps - pre_train_steps)
            # BETA = 0.5

        # Perform one step of the optimization (on the policy network)
        loss = model.optimize_model(optimizer, policy_net, target_net, replay_memory, demo_replay_memory, dqfd_loss, BATCH_SIZE=BATCH_SIZE, BETA = BETA, GAMMA=GAMMA)       
        
        # Logging step level metrics
        episode_return += reward
        episode_steps = t
        writer.add_scalar("Loss vs Total Steps (all episodes)", loss, total_steps)
        total_steps += 1

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        # print("Completed one step of soft update")
        
        # Rendering the frames and saving the model every few steps
        # env.render()
        if (total_steps % save_checkpoint) == 0:
            torch.save(policy_net.state_dict(), save_path)

        if done:
            break
        # print("--------------")

    # Logging episode level metrics
    writer.add_scalar("Num Steps vs Episode", episode_steps, i_episode)
    writer.add_scalar("Total Episode Return vs Episode", episode_return, i_episode)

writer.close()
torch.save(policy_net.state_dict(), save_path)

print('Complete')
