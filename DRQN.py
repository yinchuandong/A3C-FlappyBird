import tensorflow as tf
import numpy as np
import random
import time
import os
import sys
from netutil import *
from game.flappy_bird import FlappyBird
from replay_buffer import ReplayBuffer

INPUT_SIZE = 84
INPUT_CHANNEL = 1
ACTIONS_DIM = 2

LSTM_UNITS = 512
LSTM_MAX_STEP = 8

GAMMA = 0.99
FINAL_EPSILON = 0.1
INITIAL_EPSILON = 1.0
ALPHA = 1e-6  # the learning rate of optimizer
TAU = 0.001
UPDATE_FREQUENCY = 5  # the frequency to update target network

MAX_TIME_STEP = 10 * 10 ** 7
EPSILON_TIME_STEP = 1 * 10 ** 6  # for annealing the epsilon greedy
EPSILON_ANNEAL = float(INITIAL_EPSILON - FINAL_EPSILON) / EPSILON_TIME_STEP
BATCH_SIZE = 4
REPLAY_MEMORY = 5000

CHECKPOINT_DIR = 'tmp_drqn/checkpoints'
LOG_FILE = 'tmp_drqn/log'


class Network(object):

    def __init__(self, scope):
        # input layer
        s = tf.placeholder('float', shape=[None, INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL], name='s')

        # hidden conv layer
        W_conv1 = weight_variable([8, 8, INPUT_CHANNEL, 16])
        b_conv1 = bias_variable([16])
        h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)

        W_conv2 = weight_variable([4, 4, 16, 32])
        b_conv2 = bias_variable([32])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

        W_conv3 = weight_variable([3, 3, 32, 64])
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

        h_conv3_out_size = np.prod(h_conv3.get_shape().as_list()[1:])
        print 'conv flat size:', h_conv3_out_size
        h_conv3_flat = tf.reshape(h_conv3, [-1, h_conv3_out_size])

        W_fc1 = weight_variable([h_conv3_out_size, LSTM_UNITS])
        b_fc1 = bias_variable([LSTM_UNITS])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # reshape to fit lstm (batch_size, timestep, LSTM_UNITS)
        self.timestep = tf.placeholder(dtype=tf.int32)
        self.batch_size = tf.placeholder(dtype=tf.int32)

        h_fc1_reshaped = tf.reshape(h_fc1, [self.batch_size, self.timestep, LSTM_UNITS])
        self.lstm_cell = tf.contrib.rnn.LSTMCell(num_units=LSTM_UNITS, state_is_tuple=True)
        self.initial_lstm_state = self.lstm_cell.zero_state(self.batch_size, tf.float32)
        lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(
            self.lstm_cell,
            h_fc1_reshaped,
            initial_state=self.initial_lstm_state,
            sequence_length=self.timestep,
            time_major=False,
            dtype=tf.float32,
            scope=scope
        )
        print 'lstm shape:', lstm_outputs.get_shape()
        # shape: [batch_size*timestep, LSTM_UNITS]
        lstm_outputs = tf.reshape(lstm_outputs, [-1, LSTM_UNITS])

        streamA, streamV = tf.split(lstm_outputs, 2, axis=1)
        AW = tf.Variable(tf.random_normal([LSTM_UNITS / 2, ACTIONS_DIM]))
        VW = tf.Variable(tf.random_normal([LSTM_UNITS / 2, 1]))
        advantage = tf.matmul(streamA, AW)
        value = tf.matmul(streamV, VW)

        # readout layer: Q_value
        Q_value = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keep_dims=True))
        print 'Q shape:', Q_value.get_shape()

        self.s = s
        self.Q_value = Q_value
        return


class DRQN(object):

    def __init__(self):
        self.global_t = 0
        self.replay_buffer = ReplayBuffer(REPLAY_MEMORY)

        # q-network parameter
        self.create_network()
        self.create_minimize()

        # init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        update_target(self.session, self.target_ops)

        self.saver = tf.train.Saver(tf.global_variables())
        self.restore()

        self.epsilon = 1 - float(INITIAL_EPSILON - FINAL_EPSILON) \
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
        self.mainNet = Network(scope='main')
        self.targetNet = Network(scope='target')
        self.target_ops = update_target_graph_op(tf.trainable_variables(), TAU)
        return

    def create_minimize(self):
        self.a = tf.placeholder('float', shape=[None, ACTIONS_DIM], name='a')
        self.y = tf.placeholder('float', shape=[None], name='y')
        Q_action = tf.reduce_sum(tf.multiply(self.mainNet.Q_value, self.a), axis=1)
        self.loss = tf.reduce_mean(tf.square(self.y - Q_action))
        self.optimizer = tf.train.AdamOptimizer(ALPHA)
        self.apply_gradients = self.optimizer.minimize(self.loss)
        return

    def perceive(self, state, action, reward, next_state, terminal):
        self.global_t += 1

        self.episode_reward += reward
        if self.episode_start_time == 0.0:
            self.episode_start_time = time.time()

        if terminal or self.global_t % 10 == 0:
            living_time = time.time() - self.episode_start_time
            self.record_log(self.episode_reward, living_time)

        if terminal:
            self.episode_reward = 0.0
            self.episode_start_time = time.time()

        if self.replay_buffer.size() > BATCH_SIZE:
            self.train_Q_network()

        if self.global_t % 100000 == 0:
            self.backup()
        return

    def get_action_index(self, state, lstm_state):
        Q_value_t, lstm_state_out = self.session.run(
            [self.mainNet.Q_value, self.mainNet.lstm_state],
            feed_dict={self.mainNet.s: [state], self.mainNet.initial_lstm_state: lstm_state}
        )
        return np.argmax(Q_value_t[0]), np.max(Q_value_t[0]), lstm_state_out

    def epsilon_greedy(self, state, lstm_state):
        """
        :param state: 1x84x84x3
        """
        Q_value_t, lstm_state_out = self.session.run(
            [self.mainNet.Q_value, self.mainNet.lstm_state],
            feed_dict={
                self.mainNet.s: [state],
                self.mainNet.initial_lstm_state: lstm_state,
                self.mainNet.batch_size: 1,
                self.mainNet.timestep: 1
            })
        Q_value_t = Q_value_t[0]
        action_index = 0
        if random.random() <= self.epsilon:
            action_index = random.randrange(ACTIONS_DIM)
            print 'random-index:', action_index
        else:
            action_index = np.argmax(Q_value_t)

        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= EPSILON_ANNEAL
        max_q_value = np.max(Q_value_t)
        return action_index, max_q_value, lstm_state_out

    def train_Q_network(self):
        '''
        do backpropogation
        '''
        # len(minibatch) = BATCH_SIZE * LSTM_MAX_STEP
        
        if self.global_t % (UPDATE_FREQUENCY * 100) == 0:
            update_target(self.session, self.target_ops)

        if self.global_t % UPDATE_FREQUENCY != 0:
            return
        minibatch = self.replay_buffer.sample(BATCH_SIZE, LSTM_MAX_STEP)
        state_batch = [t[0] for t in minibatch]
        action_batch = [t[1] for t in minibatch]
        reward_batch = [t[2] for t in minibatch]
        next_state_batch = [t[3] for t in minibatch]
        terminal_batch = [t[4] for t in minibatch]

        y_batch = []
        # todo: need to feed with batch_size, timestep, lstm_state
        lstm_state_train = (np.zeros([BATCH_SIZE, LSTM_UNITS]), np.zeros([BATCH_SIZE, LSTM_UNITS]))
        Q_value_batch = self.session.run(
            self.targetNet.Q_value,
            feed_dict={
                self.targetNet.s: next_state_batch,
                self.targetNet.initial_lstm_state: lstm_state_train,
                self.targetNet.batch_size: BATCH_SIZE,
                self.targetNet.timestep: LSTM_MAX_STEP
            }
        )
        for i in range(len(state_batch)):
            terminal = terminal_batch[i]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.session.run(self.apply_gradients, feed_dict={
            self.y: y_batch,
            self.a: action_batch,
            self.mainNet.s: state_batch,
            self.mainNet.initial_lstm_state: lstm_state_train,
            self.mainNet.batch_size: BATCH_SIZE,
            self.mainNet.timestep: LSTM_MAX_STEP
        })

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
    agent = DRQN()
    env = FlappyBird()

    while True:
        episode_buffer = []
        lstm_state = (np.zeros([1, LSTM_UNITS]), np.zeros([1, LSTM_UNITS]))
        count = 0
        while not env.terminal:
            # action_id = random.randint(0, 1)
            action_id, action_q, lstm_state = agent.epsilon_greedy(
                np.reshape(env.s_t[:, :, -1], (84, 84, 1)), lstm_state)
            env.process(action_id)

            action = np.zeros(ACTIONS_DIM)
            action[action_id] = 1
            state = np.reshape(env.s_t[:, :, -1], (84, 84, 1))
            next_state = np.reshape(env.s_t1[:, :, -1], (84, 84, 1))
            reward = env.reward
            terminal = env.terminal
            # frame skip
            if count % 3 == 0:
                episode_buffer.append((state, action, reward, next_state, terminal))
            agent.perceive(state, action, reward, next_state, terminal)
            print 'global_t:', agent.global_t, '/terminal:', terminal, '/action_q', action_q, \
                '/epsilon:', agent.epsilon

            env.update()
            if len(episode_buffer) >= 50:
                # start a new episode buffer, in case of an over-long memory
                agent.replay_buffer.add(episode_buffer)
                episode_buffer = []
                print '----------- episode buffer > 100---------'
            count += 1
        # reset the state
        env.reset()
        if len(episode_buffer) > LSTM_MAX_STEP:
            agent.replay_buffer.add(episode_buffer)
        print 'episode_buffer', len(episode_buffer)
        print 'replay_buffer.size:', agent.replay_buffer.size()
        # break
    return


if __name__ == '__main__':
    main()
