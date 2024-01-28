import random
from collections import deque, namedtuple

class ExperienceReplay:
    def __init__(self, capacity, batch_size):
        self.memory = deque(maxlen= capacity)
        self.Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'] )
        self.batch_size = batch_size

    def add_experience(self, state, action, reward, next_state, done):
        experience = self.Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        return batch

    def can_provide_sample(self):
        a = len(self.memory) >= self.batch_size
        return a