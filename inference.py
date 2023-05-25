import torch
import torch.optim as optim
import cv2
import numpy as np
from tqdm import tqdm
import psutil
import gym
import minerl
# from minerl.data import BufferedBatchIter

import model # Import the classes and functions defined in model.py
from utils import stack_observations, pad_state
from actions import actions as action_list
# from demo_sampling import sample_demo_batch

# from torch.utils.tensorboard import SummaryWriter

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
# BATCH_SIZE = 32
GAMMA = 0.99
EPS = 0.00 # Setting epsilon to zero implies that the agent always exploits the learnt policy
# TAU = 0.005
# LR = 1e-4
# num_episodes = 5
num_steps = 500
# save_checkpoint = 500 # save the model after these many steps
# pre_train_steps = int(2*num_steps)

# Choosing the saved model's architecture (the input shape to the model varies as per its architecture). For now, we only consider the duelling net architecture (as the simple one was only for testing)
# architecture = "simple"  
architecture = "duelling_net"

# Select a learnt model from the saved_models directory
save_path = "saved_models/'frame_stack:4_|batch_size:32_|gamma:0.99_|eps:0.01_|tau:0.005_|lr:0.0001_|episodes:5_|steps:1500_|run:Test_Run_4.pt"

# Creating the environment (this may take a few minutes) and setting up the data sampling iterator
env = gym.make('MineRLTreechop-v0')
print("Gym.make done")

# n_observation_feats =  FRAME_STACK * 64 * 64 #  64 * 64 * 3 * FRAME_STACK 
# n_actions = 15
# print(f"num observation features: {n_observation_feats}")
# print(f"num actions: {n_actions}")


# Initialize the environment and get it's state
obs = env.reset()
print("Reset Successful")
obs_gray = cv2.cvtColor(obs['pov'], cv2.COLOR_BGR2GRAY)
# Stacking observations together to form a state
state = stack_observations(obs_gray, FRAME_STACK)
# print(f"First state shape: {state.shape}")


loop = tqdm(range(num_steps))
for t in loop:
    # loop.set_description(f"Episode {i_episode} Steps | CPU {psutil.cpu_percent()} | RAM {psutil.virtual_memory().percent}")
    loop.set_description(f"Episode Steps | CPU {psutil.cpu_percent()} | RAM {psutil.virtual_memory().percent}")

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

    # Move to the next state
    state = next_state

    env.render()

    if done:
	    break
	# print("--------------")


print('Complete')