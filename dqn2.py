import tensorflow as tf
import numpy as np
import random
import time
import os
from collections import deque
from netutil import *
from game.flappy_bird import FlappyBird

INPUT_SIZE = 80
INPUT_CHANNEL = 4
ACTIONS_DIM = 2

GAMMA = 0.99
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.0001

ALPHA = 1e-6  # the learning rate of optimizer

MAX_TIME_STEP = 10 * 10 ** 7
EPSILON_TIME_STEP = 1 * 10 ** 6  # for annealing the epsilon greedy
EPSILON_ANNEAL = float(INITIAL_EPSILON - FINAL_EPSILON) / EPSILON_TIME_STEP
REPLAY_MEMORY = 50000
BATCH_SIZE = 32

CHECKPOINT_DIR = 'tmp_dqn2/checkpoints'
LOG_FILE = 'tmp_dqn2/log'


class DQN(object):

    def __init__(self):
        self.global_t = 0
        self.replay_buffer = deque(maxlen=REPLAY_MEMORY)

        # q-network parameter
        self.create_network()
        self.create_minimize()

        # init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(tf.global_variables())
        self.restore()
        self.epsilon = INITIAL_EPSILON - float(INITIAL_EPSILON - FINAL_EPSILON) \
            * min(self.global_t, EPSILON_TIME_STEP) / float(EPSILON_TIME_STEP)

        # for recording the log into tensorboard
        self.time_input = tf.placeholder(tf.float32)
        self.reward_input = tf.placeholder(tf.float32)
        tf.summary.scalar('living_time', self.time_input)
        tf.summary.scalar('reward', self.reward_input)
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(LOG_FILE, self.session.graph)

        self.episode_start_time = 0.0
        self.episode_reward = 0.0
        return

    def create_network(self):
        # input layer
        s = tf.placeholder('float', shape=[None, INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL], name='s')

        # hidden conv layer
        W_conv1 = weight_variable([8, 8, INPUT_CHANNEL, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)

        h_poo1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([4, 4, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_poo1, W_conv2, 2) + b_conv2)

        W_conv3 = weight_variable([3, 3, 64, 64])
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

        h_conv3_out_size = np.prod(h_conv3.get_shape().as_list()[1:])
        print h_conv3_out_size
        h_conv3_flat = tf.reshape(h_conv3, [-1, h_conv3_out_size])

        W_fc1 = weight_variable([h_conv3_out_size, 512])
        b_fc1 = bias_variable([512])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # readout layer: Q_value
        W_fc2 = weight_variable([512, ACTIONS_DIM])
        b_fc2 = bias_variable([ACTIONS_DIM])
        Q_value = tf.matmul(h_fc1, W_fc2) + b_fc2

        self.s = s
        self.Q_value = Q_value
        return

    def create_minimize(self):
        self.a = tf.placeholder('float', shape=[None, ACTIONS_DIM])
        self.y = tf.placeholder('float', shape=[None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.a), reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(self.y - Q_action))
        self.optimizer = tf.train.AdamOptimizer(ALPHA)
        self.apply_gradients = self.optimizer.minimize(self.loss)
        return

    def perceive(self, state, action, reward, next_state, terminal):
        self.global_t += 1

        self.replay_buffer.append((state, action, reward, next_state, terminal))

        self.episode_reward += reward
        if self.episode_start_time == 0.0:
            self.episode_start_time = time.time()

        if terminal or self.global_t % 600 == 0:
            living_time = time.time() - self.episode_start_time
            self.record_log(self.episode_reward, living_time)

        if terminal:
            self.episode_reward = 0.0
            self.episode_start_time = time.time()

        if len(self.replay_buffer) > BATCH_SIZE * 4:
            self.train_Q_network()
        return

    def get_action_index(self, state):
        Q_value_t = self.session.run(self.Q_value, feed_dict={self.s: state})[0]
        return np.argmax(Q_value_t), np.max(Q_value_t)

    def epsilon_greedy(self, state):
        """
        :param state: 1x84x84x3
        """
        Q_value_t = self.session.run(self.Q_value, feed_dict={self.s: [state]})[0]
        action_index = 0
        if random.random() <= self.epsilon:
            action_index = random.randrange(ACTIONS_DIM)
        else:
            action_index = np.argmax(Q_value_t)

        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= EPSILON_ANNEAL
        max_q_value = np.max(Q_value_t)
        return action_index, max_q_value

    def train_Q_network(self):
        '''
        do backpropogation
        '''
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [t[0] for t in minibatch]
        action_batch = [t[1] for t in minibatch]
        reward_batch = [t[2] for t in minibatch]
        next_state_batch = [t[3] for t in minibatch]
        terminal_batch = [t[4] for t in minibatch]

        y_batch = []
        Q_value_batch = self.session.run(self.Q_value, feed_dict={self.s: next_state_batch})
        for i in range(BATCH_SIZE):
            terminal = terminal_batch[i]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.session.run(self.apply_gradients, feed_dict={
            self.y: y_batch,
            self.a: action_batch,
            self.s: state_batch
        })

        if self.global_t % 100000 == 0:
            self.backup()
        return

    def record_log(self, reward, living_time):
        '''
        record the change of reward into tensorboard log
        '''
        summary_str = self.session.run(self.summary_op, feed_dict={
            self.reward_input: reward,
            self.time_input: living_time
        })
        self.summary_writer.add_summary(summary_str, self.global_t)
        return

    def restore(self):
        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("checkpoint loaded:", checkpoint.model_checkpoint_path)
            tokens = checkpoint.model_checkpoint_path.split("-")
            # set global step
            self.global_t = int(tokens[1])
            print(">>> global step set: ", self.global_t)
        else:
            print("Could not find old checkpoint")
        return

    def backup(self):
        if not os.path.exists(CHECKPOINT_DIR):
            os.mkdir(CHECKPOINT_DIR)

        self.saver.save(self.session, CHECKPOINT_DIR + '/' + 'checkpoint', global_step=self.global_t)
        return


def main():
    '''
    the function for training
    '''
    agent = DQN()
    game = FlappyBird()
    game.reset()

    while agent.global_t < MAX_TIME_STEP:
        action_id, action_q = agent.epsilon_greedy(game.s_t)
        game.process(action_id)
        action = np.zeros(ACTIONS_DIM)
        action[action_id] = 1
        agent.perceive(game.s_t, action, game.reward, game.s_t1, game.terminal)

        if agent.global_t % 10 == 0:
            print 'global_t:', agent.global_t, '/ epsilon:', agent.epsilon, '/ terminal:', game.terminal, \
                '/ action:', action_id, '/ reward:', game.reward, '/ q_value:', action_q

        if game.terminal:
            game.reset()
        # s_t <- s_t1
        game.update()

    return


if __name__ == '__main__':
    print 'dd'
    main()
