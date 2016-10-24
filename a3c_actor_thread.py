import tensorflow as tf
import numpy as np
import random

from accum_trainer import AccumTrainer
from a3c_network import A3CFFNetwork
from config import *
from game_state import GameState


class A3CActorThread(object):

    def __init__(self,
                 thread_index,
                 global_network,
                 initial_learning_rate,
                 learning_rate_input,
                 optimizer,
                 max_global_time_step,
                 device
                 ):

        self.thread_index = thread_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step

        self.local_network = A3CFFNetwork(STATE_DIM, STATE_CHN, ACTION_DIM, device)
        self.local_network.create_loss(ENTROPY_BETA)
        self.trainer = AccumTrainer(device)
        self.trainer.create_minimize(self.local_network.get_total_loss(), self.local_network.get_vars())
        self.accum_gradients = self.trainer.accumulate_gradients()
        self.reset_gradients = self.trainer.reset_gradients()

        self.apply_gradients = optimizer.apply_gradients(
            zip(self.trainer.get_accum_grad_list(), global_network.get_vars()))
        self.sync = self.local_network.sync_from(global_network)
        self.game_state = GameState()
        self.local_t = 0
        self.initial_learning_rate = initial_learning_rate
        self.episode_reward = 0
        self.prev_local_t = 0
        return

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * \
            (self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def choose_action(self, policy_output):
        values = []
        sum = 0.0
        for rate in policy_output:
            sum += rate
            values.append(sum)

        r = random.random() * sum
        for i in range(len(values)):
            if values[i] >= r:
                return i
        return len(values) - 1


if __name__ == '__main__':
    a = [1, 2, 3]
    b = ['a', 'b', 'c']
    print zip(a, b)
    print 'he'
