import torch
import torch.optim as optim
import cv2
import numpy as np
from tqdm import tqdm
import psutil
import gym
import minerl

import model # Import the classes and functions defined in model.py
from utils import stack_observations, pad_state
from actions import actions as action_list


# Setting up a device
print(f"Is GPU available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

# Defining model hyper-parameters
# EPS is the epsilon greedy exploration probability
num_episodes = 7 # Number of episodes of inference to run
FRAME_STACK = 4
EPS = 0.00 # Setting epsilon to zero implies that the agent always exploits the learnt policy
num_steps = 250 # Number of steps to run through during inference
n_actions = 15
# Select a learnt model from the saved_models directory
save_path = "saved_models/frame_stack:4_|batch_size:32_|gamma:0.99_|eps:0.01_|tau:0.005_|lr:0.0001_|episodes:5_|steps:1500_|run:Test_Run_4.pt"

# Choosing the saved model's architecture (the input shape to the model varies as per its architecture). For now, we only consider the duelling net architecture (as the simple one was only for testing)
# architecture = "simple"  
architecture = "duelling_net"

if architecture == "duelling_net":
    # Defining the duelling network Q networks
    policy_net = model.dueling_net(n_actions, FRAME_STACK).to(device)
    # policy_net = policy_net.float()
    policy_net.load_state_dict(torch.load(save_path))


# Creating the environment (this may take a few minutes) and setting up the data sampling iterator
env = gym.make('MineRLTreechop-v0')
print("Gym.make done")


for i_episode in range(num_episodes):
	# Initialize the environment and get it's state
	obs = env.reset()
	print("Reset Successful")
	obs_gray = cv2.cvtColor(obs['pov'], cv2.COLOR_BGR2GRAY)
	# Stacking observations together to form a state
	state = stack_observations(obs_gray, FRAME_STACK)
	# print(f"First state shape: {state.shape}")


	loop = tqdm(range(num_steps))
	for t in loop:
	    loop.set_description(f"Episode {i_episode} Steps | CPU {psutil.cpu_percent()} | RAM {psutil.virtual_memory().percent}")

	    # Some shape transformations to align the state shape with the model's expected input shape
	    temp = torch.tensor(state, dtype=torch.float32)
	    shape = list(temp.shape)
	    shape.insert(0,1)
	    action = model.select_action(temp.view(tuple(shape)), EPS, policy_net)

	    next_state = np.zeros(state.shape)
	    reward = 0
	    done = False
	    for i in range(FRAME_STACK):
	        if not done:
	            next_observation, next_reward, done, _ = env.step(action_list[action])
	            next_obs_gray = cv2.cvtColor(next_observation['pov'], cv2.COLOR_BGR2GRAY)
	            next_state[i] = next_obs_gray
	            reward += next_reward

	    # Move to the next state
	    state = next_state

	    env.render()

	    if done:
		    break

print('Complete')