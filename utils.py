import matplotlib
import matplotlib.pyplot as plt
import numpy as np

'''
List of Plots
1. Rewards vs training steps
2. Loss vs training steps during pre-training
3. Num steps in episode vs episodes
'''

def stack_observations(observation, stack_size):
	'''
	Utility function to stack k observations together. Used to stack the first observations on resetting the environment
	'''
	output_shape = list(observation.shape)
	output_shape.insert(0,stack_size)
	output_shape = tuple(output_shape)
	stacked_observation = np.zeros(output_shape)
	for i in range(stack_size):
		stacked_observation[i] = observation

	return stacked_observation

def pad_state(state, stack_size):
	'''
	Observations are stacked on top of eachother to comply with frame stacking. The Q network returns an action based on the current state. This action is
	repeated k-1 times to generate a next_state with k observations. However, in the event that the episode ends before all k next_observations can be 
	added to the next_state, this tensor will be padded to make it have a k observations in total. This function does the padding.
	'''
	num_obserations = state.shape[0]
	output_shape = list(state.shape[1:])
	output_shape.insert(0,stack_size)
	output_shape = tuple(output_shape)
	padded_state = np.zeros(output_shape)
	padded_state[:num_obserations] = state
	for i in range(stack_size - num_obserations):
		padded_state[i+num_obserations] = state[-1]

	return padded_state


