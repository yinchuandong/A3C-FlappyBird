from collections import deque
import random
import numpy as np


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
        return

    def sample(self, batch_size, timestep):
        '''
        sample from buffer, get [batch_size][timestep]
        return a reshaped array with size: batch_size*timestep
        '''
        episode_batch = random.sample(self.buffer, batch_size)
        experience = []
        for episode in episode_batch:
            start = random.randint(0, len(episode) - timestep)
            experience.append(episode[start:start + timestep])
        experience = np.array(experience)
        return np.reshape(experience, [batch_size * timestep])

    def capacity(self):
        return self.capacity

    def size(self):
        return len(self.buffer)

    def add(self, episode_buffer):
        '''
        note: each element in replay buffer is an array, contains a series of episodes
            like: [(s, a, r, d, s1)]
        '''
        self.buffer.append(episode_buffer)
        return

    def get_recent_state(self):
        return self.buffer[-1][-1]


if __name__ == '__main__':
    rp = ReplayBuffer(10000)
