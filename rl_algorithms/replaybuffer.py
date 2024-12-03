import random
import numpy as np

import numpy as np

class ReplayBuffer:
    def __init__(self, max_size):
        """
        Replay buffer for storing objects like traj_batch.
        """
        self.max_size = max_size
        self.buffer = []
        self.mem_cntr = 0  # Counter to track the number of stored items

    def store_object(self, obj):
        """
        Store an object in the buffer and manage capacity.
        """
        if len(self.buffer) < self.max_size:
            self.buffer.append(obj)  # Append if there's space
        else:
            self.buffer[self.mem_cntr % self.max_size] = obj  # Overwrite oldest memory

        self.mem_cntr += 1

    def sample_objects(self):
        """
        Randomly pick a single TrajBatchAmp object from the buffer.
        """
        if len(self.buffer) == 0:
            raise ValueError("The buffer is empty. Cannot sample objects.")

        # Randomly pick a single index
        index = np.random.randint(0, len(self.buffer))

        # Return the object at the selected index
        return self.buffer[index]

    def size(self):
        """
        Return the current number of objects in the buffer.
        """
        return len(self.buffer)

    def is_full(self):
        """
        Check if the buffer is at maximum capacity.
        """
        return len(self.buffer) == self.max_size