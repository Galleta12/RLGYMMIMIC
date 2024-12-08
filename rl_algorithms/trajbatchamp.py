import numpy as np


class TrajBatchAmp:

    def __init__(self, memory_list):
        memory = memory_list[0]
        for x in memory_list[1:]:
            memory.append(x)
        self.batch = zip(*memory.sample())
        self.states = np.stack(next(self.batch))
        self.actions = np.stack(next(self.batch))
        self.terminates = np.stack(next(self.batch))
        self.next_states = np.stack(next(self.batch))
        self.rewards = np.stack(next(self.batch))
        self.exps = np.stack(next(self.batch))
        self.amp_features = np.stack(next(self.batch))
        self.amp_next_features = np.stack(next(self.batch))
        self.amp_states = np.stack(next(self.batch))
        self.amp_next_states = np.stack(next(self.batch))
    def get_shapes(self):
        return {
            'states': self.states.shape,
            'actions': self.actions.shape,
            'terminates': self.terminates.shape,
            'next_states': self.next_states.shape,
            'rewards': self.rewards.shape,
            'exps': self.exps.shape,
            'amp_features': self.amp_features.shape,
            'amp_next_features': self.amp_next_features.shape,
            'amp_states': self.amp_states.shape,
            'amp_next_states': self.amp_next_states.shape
        }
        
class TrajBatchAmpReplay:
    def __init__(self, memory_data):
        """
        Initialize a TrajBatchAmp object from a batch of transitions.
        :param memory_data: A dictionary of fields with batched data.
        """
        self.states = memory_data['states']
        self.actions = memory_data['actions']
        self.terminates = memory_data['terminates']
        self.next_states = memory_data['next_states']
        self.rewards = memory_data['rewards']
        self.exps = memory_data['exps']
        self.amp_features = memory_data['amp_features']
        self.amp_next_features = memory_data['amp_next_features']
        self.amp_states = memory_data['amp_states']
        self.amp_next_states = memory_data['amp_next_states']

    def get_shapes(self):
        """
        Get the shapes of all fields.
        :return: A dictionary of shapes for each field.
        """
        return {
            'states': self.states.shape,
            'actions': self.actions.shape,
            'terminates': self.terminates.shape,
            'next_states': self.next_states.shape,
            'rewards': self.rewards.shape,
            'exps': self.exps.shape,
            'amp_features': self.amp_features.shape,
            'amp_next_features': self.amp_next_features.shape,
            'amp_states': self.amp_states.shape,
            'amp_next_states': self.amp_next_states.shape
        }