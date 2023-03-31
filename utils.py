import matplotlib
import matplotlib.pyplot as plt

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
	output_shape = tuple(list(observation.shape).insert(0,stack_size))
	stacked_observation = np.zeros(output_shape)
	for i in range(stack_size):
		stacked_observation[i] = observation

	return stacked_observation
