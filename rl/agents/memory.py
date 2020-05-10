import torch as T

class ExperienceReplayBuffer(object):
    def __init__(self, capacity):
        # Buffer of [N, (state, action, state_next, reward)]
        self.capacity = capacity
        self.memory = T.zeros((capacity, 4))
        self.full = False
        
        # Inital memory index in zero since memory is empty
        self.idx = 0

    def store(self, state, action, state_next, reward):
        """Stores a transition: T:(s, a, s', r)"""
        # Storing transition
        self.memory[self.idx] = T.tensor([state, action, state_next, reward])

        # Updating index to next following mem position
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == self.capacity:
            self.full = True

    def sample(self, batch_size):
        """Sample batch_size items from the memory buffer"""
        # Getting random indexes of length batch_size
        if self.full:
            # Randomize index for all memory capacity
            idx_sample = T.randperm(self.capacity)[:batch_size]
        else:
            # Randomize index only for stored idx
            idx_sample = T.randperm(self.idx)[:batch_size]
        
        # Getting the samples : [B, (s, a, s', r)]
        memory_sample = self.memory[idx_sample]
        return memory_sample