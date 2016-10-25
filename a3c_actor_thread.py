import tensorflow as tf
import numpy as np
import random

from accum_trainer import AccumTrainer
from a3c_network import A3CFFNetwork
from config import *
from game.game_state import GameState


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
        self.game_state.reset()

        self.local_t = 0
        self.initial_learning_rate = initial_learning_rate
        self.episode_reward = 0.0
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

    def process(self, sess, global_t):
        states = []
        actions = []
        rewards = []
        values = []

        terminal_end = False

        sess.run(self.reset_gradients)
        # copy weight from global network
        sess.run(self.sync)
        start_local_t = self.local_t

        for i in range(LOCAL_T_MAX):
            policy_, value_ = self.local_network.run_policy_and_value(sess, self.game_state.s_t)
            # print 'policy=', policy_
            # print 'value=', value_
            actionId = self.choose_action(policy_)

            states.append(self.game_state.s_t)
            actions.append(actionId)
            values.append(value_)

            self.game_state.process(actionId)
            reward = self.game_state.reward
            terminal = self.game_state.terminal

            self.episode_reward += reward
            rewards.append(np.clip(reward, -1.0, 1.0))

            self.local_t += 1

            # s_t1 -> s_t
            self.game_state.update()

            if terminal:
                terminal_end = True
                self.episode_reward = 0.0
                self.game_state.reset()
                break
        # -----------end of batch (LOCAL_T_MAX)--------------------

        R = 0.0
        if not terminal_end:
            R = self.local_network.run_value(sess, self.game_state.s_t)
        print ('global_t: %d, R: %f') % (global_t, R)

        states.reverse()
        actions.reverse()
        rewards.reverse()
        values.reverse()

        batch_state = []
        batch_action = []
        batch_td = []
        batch_R = []

        for (ai, ri, si, Vi) in zip(actions, rewards, states, values):
            R = ri + GAMMA * R
            td = R - Vi
            action = np.zeros([ACTION_DIM])
            action[ai] = 1

            batch_state.append(si)
            batch_action.append(action)
            batch_td.append(td)
            batch_R.append(R)

        sess.run(self.accum_gradients, feed_dict={
            self.local_network.state_input: batch_state,
            self.local_network.action_input: batch_action,
            self.local_network.td: batch_td,
            self.local_network.R: batch_R
        })

        cur_learning_rate = self._anneal_learning_rate(global_t)
        sess.run(self.apply_gradients, feed_dict={
            self.learning_rate_input: cur_learning_rate
        })

        diff_local_t = self.local_t - start_local_t
        return diff_local_t


if __name__ == '__main__':
    game_state = GameState()
    game_state.process(1)
    print np.shape(game_state.s_t)


