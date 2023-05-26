import numpy as np
from copy import deepcopy
import random

###### MONKEY PATCHING THE BUFFERED BATCH  ITER CLASS ########

''' 
Modifying the optionally_fill_buffer method to fill up the buffer with only one trajectory whenever its size is lower than the batch_size. 

Also removing shuffling the data_buffer. This is done because DQfD uses frame stacking which assumes that the transitions in a single stack are all sequential. 
However, this stack is created using the get_batch method which simply loops through the data_buffer. Shuffling the data_buffer would causes the frame stack 
to stop being sequential.  
'''
def optionally_fill_buffer_patch(self, batch_size):
    """
    This method is run after every batch, but only actually executes a buffer
    refill and re-shuffle if more data is needed
    """


    # Add trajectories to the buffer if the remaining space is
    # greater than our anticipated trajectory size (in the form of the empirical average)

    if len(self.data_buffer) < batch_size:

        if len(self.available_trajectories) == 0:
            return

        # print("loading in a trajectory from the shuffled available_trajectories")
        traj_to_load = self.available_trajectories.pop()
        data_loader = self.data_pipeline.load_data(traj_to_load)
        traj_len = 0
        for data_tuple in data_loader:
            traj_len += 1
            # self.data_buffer.append(data_tuple)
            self.data_buffer.insert(0, data_tuple) # Changing append to insert at index 0 since the get_batch method uses pop(). This way when pop() in a loop, it returns sequential frames

        self.traj_sizes.append(traj_len)
        self.avg_traj_size = np.mean(self.traj_sizes)
        # random.shuffle(self.data_buffer)


'''
Modifying the buffered_batch_iter method to stop loading all trajectories in memory as a data_buffer. This takes up too much memory and causes the Minecraft server 
to die as a result of memory overload. This patch instead only loads up one trajectory at a time and refills the buffer whenever it has fewer than FRAME_STACK number
of transitions.
'''
def buffered_batch_iter_patch(self, batch_size):
    """
    The actual generator method that returns batches. You can specify either
    a desired number of batches, or a desired number of epochs, but not both,
    since they might conflict.

    ** You must specify one or the other **

    Args:
        batch_size: The number of transitions/timesteps to be returned in each batch. Here, a batch is a sequential stack of frames. While a batch in the main
        DQfD loop is a batch of FRAME_STACK size transitions.
        num_epochs: Optional, how many full passes through all trajectories to return
        num_batches: Optional, how many batches to return

    """

    while True:
        # Remove any checks for batch_size or num_epochs as they do not make sense in the use case for this patch

        # Refill the buffer if we need to
        # (doing this before getting batch so it'll run on the first iteration)
        self.optionally_fill_buffer(batch_size=batch_size)
        ret_batch = self.get_batch(batch_size=batch_size) # 
        # if len(self.data_buffer) < batch_size:
        if len(self.available_trajectories) == 0:
            print("All available_trajectories are used up. Resetting and reshuffling available_trajectories")

            self.available_trajectories = deepcopy(self.all_trajectories)
            random.shuffle(self.available_trajectories)

        keys = ('obs', 'act', 'reward', 'next_obs', 'done')
        yield tuple([ret_batch[key] for key in keys])


###### MONKEY PATCHING THE BUFFERED BATCH  ITER CLASS ########