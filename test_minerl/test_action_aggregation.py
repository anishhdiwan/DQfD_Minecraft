import minerl
from minerl.data import BufferedBatchIter
import numpy as np
import random

# Download the dataset before running this script
data = minerl.data.make('MineRLTreechop-v0')
iterator = BufferedBatchIter(data)
demo_replay_memory = iterator.buffered_batch_iter(batch_size=2, num_epochs=1)

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






for i in range(5):
	current_states, actions, rewards, next_states, dones = next(demo_replay_memory)
	print(f'actions: {actions}')
	print(f'rewards: {rewards}')
	print('----------')

	aggregate_reward = np.sum(rewards)
	aggregate_action = get_aggregate_action(actions)
	print(f'aggregate action: {aggregate_action}')
	print(f'aggregate reward: {aggregate_reward}')
	print('################')
