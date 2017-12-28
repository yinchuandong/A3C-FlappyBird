import tensorflow as tf
import numpy as np
import random
import time

from a3c_network import A3CFFNetwork, A3CLSTMNetwork
from config import *
from game.game_state import GameState


def timestamp():
    return time.time()


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

        if USE_LSTM:
            self.local_network = A3CLSTMNetwork(STATE_DIM, STATE_CHN, ACTION_DIM, device, thread_index)
        else:
            self.local_network = A3CFFNetwork(STATE_DIM, STATE_CHN, ACTION_DIM, device, thread_index)
        self.local_network.create_loss(ENTROPY_BETA)
        self.gradients = tf.gradients(self.local_network.total_loss, self.local_network.get_vars())

        clip_accum_grads = [tf.clip_by_norm(accum_grad, 10.0) for accum_grad in self.gradients]
        self.apply_gradients = optimizer.apply_gradients(zip(clip_accum_grads, global_network.get_vars()))
        # self.apply_gradients = optimizer.apply_gradients(zip(self.gradients, global_network.get_vars()))

        self.sync = self.local_network.sync_from(global_network)

        self.game_state = GameState(thread_index)

        self.local_t = 0
        self.initial_learning_rate = initial_learning_rate

        # for log
        self.episode_reward = 0.0
        self.episode_start_time = 0.0
        self.prev_local_t = 0
        return

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * \
            (self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def choose_action(self, policy_output):
        return np.random.choice(range(len(policy_output)), p=policy_output)

    def _record_log(self, sess, global_t, summary_writer, summary_op, reward_input, reward, time_input, living_time):
        summary_str = sess.run(summary_op, feed_dict={
            reward_input: reward,
            time_input: living_time
        })
        summary_writer.add_summary(summary_str, global_t)
        summary_writer.flush()
        return

    def _discount_accum_reward(self, rewards, running_add=0.0, gamma=0.99):
        """ discounted the reward using gamma
        """
        discounted_r = np.zeros_like(rewards, dtype=np.float32)
        for t in reversed(range(len(rewards))):
            running_add = rewards[t] + running_add * gamma
            discounted_r[t] = running_add

        return list(discounted_r)

    def process(self, sess, global_t, summary_writer, summary_op, reward_input, time_input):
        batch_state = []
        batch_action = []
        batch_reward = []

        terminal_end = False
        # reduce the influence of socket connecting time
        if self.episode_start_time == 0.0:
            self.episode_start_time = timestamp()

        # copy weight from global network
        sess.run(self.sync)

        start_local_t = self.local_t
        if USE_LSTM:
            start_lstm_state = self.local_network.lstm_state_out

        for i in range(LOCAL_T_MAX):
            policy_ = self.local_network.run_policy(sess, self.game_state.s_t)
            if self.thread_index == 0 and self.local_t % 1000 == 0:
                print 'policy=', policy_

            action_id = self.choose_action(policy_)

            action_onehot = np.zeros([ACTION_DIM])
            action_onehot[action_id] = 1
            batch_state.append(self.game_state.s_t)
            batch_action.append(action_onehot)

            self.game_state.process(action_id)
            reward = self.game_state.reward
            terminal = self.game_state.terminal

            self.episode_reward += reward
            batch_reward.append(np.clip(reward, -1.0, 1.0))

            self.local_t += 1

            # s_t1 -> s_t
            self.game_state.update()

            if terminal:
                terminal_end = True
                episode_end_time = timestamp()
                living_time = episode_end_time - self.episode_start_time

                self._record_log(sess, global_t, summary_writer, summary_op,
                                 reward_input, self.episode_reward, time_input, living_time)

                print ("global_t=%d / reward=%.2f / living_time=%.4f") % (global_t, self.episode_reward, living_time)

                # reset variables
                self.episode_reward = 0.0
                self.episode_start_time = episode_end_time
                self.game_state.reset()
                if USE_LSTM:
                    self.local_network.reset_lstm_state()
                break
            # log
            if self.local_t % 40 == 0:
                living_time = timestamp() - self.episode_start_time
                self._record_log(sess, global_t, summary_writer, summary_op,
                                 reward_input, self.episode_reward, time_input, living_time)
        # -----------end of batch (LOCAL_T_MAX)--------------------

        R = 0.0
        if not terminal_end:
            R = self.local_network.run_value(sess, self.game_state.s_t)
        # print ('global_t: %d, R: %f') % (global_t, R)

        batch_value = self.local_network.run_batch_value(sess, batch_state, start_lstm_state)
        batch_R = self._discount_accum_reward(batch_reward, R, GAMMA)
        batch_td = np.array(batch_R) - np.array(batch_value)
        cur_learning_rate = self._anneal_learning_rate(global_t)

        # print("=" * 60)
        # print(batch_value)
        # print(self.local_network.run_batch_value(sess, batch_state, start_lstm_state))
        # print("=" * 60)
        # import sys
        # sys.exit()

        if USE_LSTM:
            sess.run(self.apply_gradients, feed_dict={
                self.local_network.state_input: batch_state,
                self.local_network.action_input: batch_action,
                self.local_network.td: batch_td,
                self.local_network.R: batch_R,
                self.local_network.step_size: [len(batch_state)],
                self.local_network.initial_lstm_state: start_lstm_state,
                self.learning_rate_input: cur_learning_rate
            })
        else:
            sess.run(self.apply_gradients, feed_dict={
                self.local_network.state_input: batch_state,
                self.local_network.action_input: batch_action,
                self.local_network.td: batch_td,
                self.local_network.R: batch_R,
                self.learning_rate_input: cur_learning_rate
            })

        diff_local_t = self.local_t - start_local_t
        return diff_local_t


if __name__ == '__main__':
    # game_state = GameState()
    # game_state.process(1)
    # print np.shape(game_state.s_t)
    print timestamp()
    print time.time()
