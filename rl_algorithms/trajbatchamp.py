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