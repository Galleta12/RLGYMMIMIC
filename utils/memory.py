import numpy as np

class Memory:
    """ Memory class for storing the experiences of the agent
    """
    def __init__(self):
        self._reset = True
        
    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.predictions = []
        self.terminateds = []
        self.truncateds = []
        self.next_state = []
        self.exps = []
        self.infos = []
        self.masks = []

        self._reset = False

    def __len__(self):
        return len(self.rewards)
    
    @property
    def score(self):
        return np.sum(self.rewards)

    def append(self, state, action, reward, prediction, terminated, truncated, next_state, mask,exp,info: dict={}):
        #be ready to store new experience
        if self._reset:
            self.reset()
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.predictions.append(prediction)
        self.terminateds.append(terminated)
        self.truncateds.append(truncated)
        self.next_state.append(next_state)
        self.masks.append(mask)
        self.exps.append(exp)
        self.infos.append(info)

    def get(self, reset=False):
        if reset:
            self._reset = reset

        return self.states, self.actions, self.rewards, self.predictions, self.terminateds, self.truncateds, self.next_state, self.masks,self.exps,self.infos
    
    @property
    def done(self):
        return self.terminateds[-1] or self.truncateds[-1]

class MemoryManager:
    """ Memory class for storing the experiences of the agent in multiple environments
    """
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.memory = [Memory() for _ in range(num_envs)]

    def append(self, *kwargs):
        data = list(zip(*kwargs))
        for i in range(self.num_envs):
            self.memory[i].append(*data[i])

    def __getitem__(self, index: int):
        return self.memory[index]

    def done_indices(self):
        return [env for env in range(self.num_envs) if self.memory[env].done]
    
    
    
class TrajBatch:
    def __init__(self, memory_manager):
        all_states = []
        all_actions = []
        all_masks = []
        all_next_states = []
        all_rewards = []
        all_exps = []
        all_infos = []

        for memory in memory_manager.memory:
            states, actions, rewards, predictions, terminateds, truncateds, next_states, masks, exps, infos = memory.get(reset=False)
            
            all_states.extend(states)
            all_actions.extend(actions)
            all_masks.extend(masks)
            all_next_states.extend(next_states)  # Ensure each next state is handled
            all_rewards.extend(rewards)
            all_exps.extend(exps)
            all_infos.extend(infos)

        self.states = np.array(all_states)
        self.actions = np.array(all_actions)
        self.masks = np.array(all_masks)
        self.next_states = np.array(all_next_states)
        self.rewards = np.array(all_rewards)
        self.exps = np.array(all_exps)
        self.infos = all_infos  # Infos are usually dictionaries or complex structures, so not stacked

    def get_shapes(self):
        return {
            'states': self.states.shape,
            'actions': self.actions.shape,
            'masks': self.masks.shape,
            'next_states': self.next_states.shape,
            'rewards': self.rewards.shape,
            'exps': self.exps.shape
        }