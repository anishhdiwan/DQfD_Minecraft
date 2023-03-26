import minerl
from minerl.data import BufferedBatchIter
import numpy as np
import random

# Download the dataset before running this script
data = minerl.data.make('MineRLTreechop-v0')
iterator = BufferedBatchIter(data)
demo_replay_memory = iterator.buffered_batch_iter(batch_size=2, num_epochs=1)

basic_actions = {'forward', 'back', 'left', 'right', 'attack', 'jump', 'camera'}
action_combos = [{'forward', 'left'}, {'forward', 'right'}, {'forward', 'jump'}, {'forward', 'attack'}]


def get_aggregate_action(actions, cam_threshold=2.0):
	actions.pop('sneak')
	actions.pop('sprint')
	aggregate_action = actions
	for key in aggregate_action.keys():
		if not key=='camera':
			aggregate_action[key] = np.sum(actions[key], axis=0)
		else:
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


	# print(aggregate_action)
	# print(aggregate_reward)

	cam = aggregate_action.pop('camera')
	max_idx = [i for i, x in enumerate(cam) if x == max(cam)]
	cam_dir = random.choice(max_idx) # 0,1,2,3 corresponds to l,r,u,d

	# aggregate_action = dict(sorted(aggregate_action.items(), key=lambda item: item[1]))
	aggregate_action['camera'] = [max(cam) ,cam_dir]

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

	# print(aggregate_action)
	return aggregate_action




def map_aggregate_action(aggregate_action):
	'''
	Function to map an aggregate action to one of the agent's available actions 
	'''

	if len(aggregate_action.keys()) == 0:
		action = 'noop'
	elif len(aggregate_action.keys()) == 1:
		if aggregate_action.keys()[0] in basic_actions:
			action = aggregate_action.keys()[0]
	elif len(aggregate_action.keys()) == 2:
		if set(aggregate_action.keys()) in action_combos:
			action = set(aggregate_action.keys())
		else:
			max_idx = [i for i, x in enumerate(aggregate_action.values()) if x == max(aggregate_action.values())]
			action = aggregate_action.keys()[random.choice(max_idx)]

	elif len(aggregate_action.keys()) > 2:
		action_pairs = list(combinations(aggregate_action.keys(), 2))
		max_occurences = 0
		action = None
		for pair in action_pairs:
			if set(pair) in action_combos:
				if aggregate_action[pair[0]] + aggregate_action[pair[1]] > max_occurences: 
					action = set(pair)
					max_occurences = aggregate_action[pair[0]] + aggregate_action[pair[1]]
			else:
				max_idx = [i for i, x in enumerate(aggregate_action.values()) if x == max(aggregate_action.values())]
				action = aggregate_action.keys()[random.choice(max_idx)]


	return action






for i in range(5):
	current_states, actions, rewards, next_states, dones = next(demo_replay_memory)
	# print(f'actions: {actions}')
	# print(f'rewards: {rewards}')
	# print('----------')

	aggregate_reward = np.sum(rewards)
	aggregate_action = get_aggregate_action(actions)
	print(f'aggregate action: {aggregate_action}')
	print(f'aggregate reward: {aggregate_reward}')
	print('################')


	agent_action = map_aggregate_action(aggregate_action)
	print(agent_action)


