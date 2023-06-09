import minerl
from minerl.data import BufferedBatchIter
import numpy as np
import random
from itertools import combinations

import sys
sys.path.append('../')
from actions import action_names

# Download the dataset before running this script
data = minerl.data.make('MineRLTreechop-v0')
iterator = BufferedBatchIter(data)
demo_replay_memory = iterator.buffered_batch_iter(batch_size=2, num_epochs=1)

''' 
The mineRL framework models actions as dictionaries of individual actions. Player recorded demonstration data has multiple 
combinations of actions. The number of feasible combinations is too high and this would make it very hard for the agent
to generalize. Instead, we limit the agent to a smaller set of possible actions and their combinations. These basic actions
and their combinations are listed below. While training, we use frame skipping. Hence, one state is a combination of k frames
and their k actions. Action aggregation combines these k actions into one and action mapping maps this combined action to one 
of the actions that the agent can perform.
'''

basic_actions = {'forward', 'back', 'left', 'right', 'attack', 'jump', 'look-left', 'look-right', 'look-up', 'look-down'}
action_combos = [{'forward', 'left'}, {'forward', 'right'}, {'forward', 'jump'}, {'forward', 'attack'}]


def get_aggregate_action(actions, cam_threshold=2.0):
	'''
	Function to aggregate actions from k transitions into one combined action
	NOTE: Threshold is set to discount any micro-adjustments and only count camera movements for directional navigation
	'''
	# Removing spring and sneak from the actions dict
	actions.pop('sneak')
	actions.pop('sprint')
	aggregate_action = actions
	for key in aggregate_action.keys():
		# Sum up the occurences of all actions other than the camera movement action
		if not key=='camera':
			aggregate_action[key] = np.sum(actions[key], axis=0)
		else:
			# For the camera action, instead of simply adding up movements, we compare the movement angle to a threshold
			# The absolute maximum angle from one camera movement marks the direction of camera motion (l, r, u, d)
			# We create a list with the camera movements from all k transitions called heading
			heading = [0,0,0,0] # left, right, up, down
			for i in list(actions[key]):
				idx = np.argmax(np.abs(i))
				if abs(i[idx]) > cam_threshold:
					if idx == 0:
						# Left OR Right
						if i[idx] > 0:
							# Left
							heading[0] += 1
						else:
							# Right
							heading[1] += 1
					if idx == 1:
						# Up OR Down
						if i[idx] > 0:
							# Up
							heading[2] += 1
						else:
							# Down
							heading[3] += 1
			aggregate_action[key] = np.array(heading)
			# Set camera movement to the direction that was chosen the most often. If multiple exist then choose one randomly
			max_idx = [i for i, x in enumerate(heading) if x == max(heading)] 
			cam_dir = random.choice(max_idx) # 0,1,2,3 corresponds to l,r,u,d
			# The 'camera' key now has the max number of direction occurences and the occured direction
			aggregate_action['camera'] = [max(heading) ,cam_dir]


	# print(aggregate_action)
	# print(aggregate_reward)

	# cam = aggregate_action.pop('camera')
	# max_idx = [i for i, x in enumerate(cam) if x == max(cam)]
	# cam_dir = random.choice(max_idx) # 0,1,2,3 corresponds to l,r,u,d

	# # aggregate_action = dict(sorted(aggregate_action.items(), key=lambda item: item[1]))
	# aggregate_action['camera'] = [max(cam) ,cam_dir]


	# Popping out any action that was not chosen
	noop_list = []
	for key, value in aggregate_action.items():
		if not key=='camera':
			if value == 0:
				noop_list.append(key)
		else:
			if value[0] == 0:
				noop_list.append(key)

	for key in noop_list:
		aggregate_action.pop(key)


	# Mapping camera directions to the movement and dropping out the 'camera' key
	cam_dirs = {0:'look-left', 1:'look-right', 2:'look-up', 3:'look-down'}
	if 'camera' in aggregate_action:
		cam = aggregate_action.pop('camera')
		aggregate_action[cam_dirs[cam[1]]] = cam[0]


	# print(aggregate_action)
	return aggregate_action




def map_aggregate_action(aggregate_action):
	'''
	Function to map an aggregate action to one of the agent's available actions 
	'''

	# # Mapping camera directions to the movement and dropping out the 
	# cam_dirs = {0:'look-left', 1:'look-right', 2:'look-up', 3:'look-down'}
	# if 'camera' in aggregate_action:
	# 	cam = aggregate_action.pop('camera')
	# 	aggregate_action[cam_dirs[cam[1]]] = cam[0]

	# If empty then select no-operation action
	if len(aggregate_action.keys()) == 0:
		action = 'noop'

	# If there is only one action then pick that one 
	elif len(aggregate_action.keys()) == 1:
		if list(aggregate_action.keys())[0] in basic_actions:
			action = list(aggregate_action.keys())[0]

	# If there are two actions then check if that pair is possible. Pick the pair if it is, else pick the most occuring one
	elif len(aggregate_action.keys()) == 2:
		if set(aggregate_action.keys()) in action_combos:
			action = list(aggregate_action.keys())[0] + "_" + list(aggregate_action.keys())[1] 
		else:
			max_idx = [i for i, x in enumerate(aggregate_action.values()) if x == max(aggregate_action.values())]
			action = list(aggregate_action.keys())[random.choice(max_idx)]

	# If there are more than 2 actions then check all pairs. Pick a pair with the max total occurence count
	elif len(aggregate_action.keys()) > 2:
		action_pairs = list(combinations(aggregate_action.keys(), 2))
		max_occurences = 0
		action = None
		pair_match = False
		for pair in action_pairs:
			if set(pair) in action_combos:
				pair_match = True
				if aggregate_action[pair[0]] + aggregate_action[pair[1]] > max_occurences: 
					max_occurences = aggregate_action[pair[0]] + aggregate_action[pair[1]]
					action = pair[0] + "_" + pair[1]
		if not pair_match:
			max_idx = [i for i, x in enumerate(aggregate_action.values()) if x == max(aggregate_action.values())]
			action = list(aggregate_action.keys())[random.choice(max_idx)]


	return action


for i in range(10):
	current_states, actions, rewards, next_states, dones = next(demo_replay_memory)
	# print(f'actions: {actions}')
	# print(f'rewards: {rewards}')
	print('----------')

	aggregate_reward = np.sum(rewards)
	aggregate_action = get_aggregate_action(actions)
	print(f'aggregate action: {aggregate_action}')
	print(f'aggregate reward: {aggregate_reward}')

	agent_action = map_aggregate_action(aggregate_action)
	idx = action_names[agent_action]

	print(f'agent action: {agent_action}')
	print(f'idx: {idx}')

	print('----------')



