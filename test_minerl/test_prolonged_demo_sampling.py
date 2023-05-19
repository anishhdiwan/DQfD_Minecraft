import minerl
from minerl.data import BufferedBatchIter
from tqdm import tqdm
import sys
import psutil

sys.path.append('../')
from demo_sampling import sample_demo_batch



# Download the dataset before running this script
data = minerl.data.make('MineRLTreechop-v0')
iterator = BufferedBatchIter(data)
demo_replay_memory = iterator.buffered_batch_iter(batch_size=2, num_epochs=1) # The batch_size here refers to the number of consequtive frames


################
# Long loop
################



loop = tqdm(range(int(1500)))
for t in loop:
	loop.set_description(f"Testing Prolonged Sampling. CPU {psutil.cpu_percent()} | RAM {psutil.virtual_memory().percent}")
	# batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = sample_demo_batch(demo_replay_memory, 64, grayscale=True)

	current_states, actions, rewards, next_states, dones = next(demo_replay_memory)

# print(action)
# print('-----------')
# print(reward)
# print('-----------')
# print(done)
