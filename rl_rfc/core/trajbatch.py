import numpy as np


class TrajBatch:

    def __init__(self, memory_list):
        memory = memory_list[0]
        for x in memory_list[1:]:
            memory.append(x)
        self.batch = zip(*memory.sample())
        self.states = np.stack(next(self.batch))
        self.actions = np.stack(next(self.batch))
        self.masks = np.stack(next(self.batch))
        self.next_states = np.stack(next(self.batch))
        self.rewards = np.stack(next(self.batch))
        self.exps = np.stack(next(self.batch))
    def get_shapes(self):
        return {
            'states': self.states.shape,
            'actions': self.actions.shape,
            'masks': self.masks.shape,
            'next_states': self.next_states.shape,
            'rewards': self.rewards.shape,
            'exps': self.exps.shape
        }