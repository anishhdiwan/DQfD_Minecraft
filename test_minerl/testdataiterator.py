import minerl
from minerl.data import BufferedBatchIter

# Download the dataset before running this script
data = minerl.data.make('MineRLTreechop-v0')
iterator = BufferedBatchIter(data)

gen = iterator.buffered_batch_iter(batch_size=4, num_epochs=1)
current_state, action, reward, next_state, done = next(gen)

print(action)
print('-----------')
print(reward)
