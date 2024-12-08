import random
import numpy as np

import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, batch_size, fields):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.batch_size = batch_size
        self.fields = fields

        # Create buffers for each field using np.float64
        for field, shape in self.fields.items():
            setattr(self, field, np.zeros((max_size,) + shape, dtype=np.float32))
    
    
    def store_transition(self, items):
        """
        Store a single transition or a batch of transitions into the replay buffer.
        :param items: A dictionary of {field_name: field_value}.
                      Each field_value can be a single transition or a batch of transitions.
        """
        num_items = len(next(iter(items.values())))  # Determine if it's a batch or single transition
        
        #print('num_items',num_items)
        start_index = self.mem_cntr % self.mem_size  # Start position for storing
        end_index = (start_index + num_items) % self.mem_size  # End position for storing

        # If the data wraps around, split the storage
        for field, value in items.items():
            if end_index > start_index:
                getattr(self, field)[start_index:end_index] = value
            else:
                # Split the data between the end and the beginning
                split_index = self.mem_size - start_index
                getattr(self, field)[start_index:] = value[:split_index]
                getattr(self, field)[:end_index] = value[split_index:]

        self.mem_cntr += num_items
    
    def sample_buffer(self):
        """
        Sample a batch of transitions uniformly from the replay buffer.
        :return: A dictionary of sampled transitions.
        """
        max_mem = min(self.mem_cntr, self.mem_size)

        # Uniformly sample a batch of indices
        batch_indices = np.random.choice(max_mem, self.batch_size, replace=False)

        # Collect sampled data for all fields
        sampled_data = {field: getattr(self, field)[batch_indices] for field in self.fields}

      

        return sampled_data

    
    def ready(self):
        """
        Check if the buffer is ready for sampling.
        :return: True if the buffer has at least `batch_size` samples.
        """
        return self.mem_cntr >= self.batch_size
    
    def size(self):
        """
        Get the current size of the buffer (number of stored transitions).
        :return: The current size of the buffer.
        """
        return min(self.mem_cntr, self.mem_size)