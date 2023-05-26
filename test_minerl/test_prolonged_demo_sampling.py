import minerl
from minerl.data import BufferedBatchIter
from tqdm import tqdm
import sys
import psutil

sys.path.append('../')
from demo_sampling import sample_demo_batch
from buffered_batch_iter_patches import optionally_fill_buffer_patch, buffered_batch_iter_patch

BufferedBatchIter.optionally_fill_buffer = optionally_fill_buffer_patch
BufferedBatchIter.buffered_batch_iter = buffered_batch_iter_patch



# Download the dataset before running this script
data = minerl.data.make('MineRLTreechop-v0')
iterator = BufferedBatchIter(data, buffer_target_size=5000)
demo_replay_memory = iterator.buffered_batch_iter(batch_size=2) # The batch_size here refers to the number of consequtive frames


################
# Loop with next
################

loop = tqdm(range(int(4000)))
for t in loop:
	loop.set_description(f"Testing Prolonged Sampling. CPU {psutil.cpu_percent()} | RAM {psutil.virtual_memory().percent}")
	batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = sample_demo_batch(demo_replay_memory, 64, grayscale=True)

	# current_states, actions, rewards, next_states, dones = next(demo_replay_memory)
	# print(current_states['pov'].shape)
	# print(next_states['pov'].shape)
	# print(actions)
	# print(rewards)
	# print(dones)


################
# For loop through iterator
################
# print("---------------------")
# for j in range(3):
# 	count = 0
# 	for current_states, actions, rewards, next_states, dones in iterator.buffered_batch_iter(batch_size=2, num_epochs=1):
# 		print(f"Testing Prolonged Sampling. CPU {psutil.cpu_percent()} | RAM {psutil.virtual_memory().percent} | {count}/20000", end="\r")
# 		# print(current_states['pov'].shape)
# 		# print(next_states['pov'].shape)
# 		# print(actions)
# 		# print(rewards)
# 		# print(dones)
# 		count += 1
# 		if count == 20000:
# 			break



# print(action)
# print('-----------')
# print(reward)
# print('-----------')
# print(done)
