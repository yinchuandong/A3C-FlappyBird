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
INPUT_CHANNEL = 4
ACTIONS_DIM = 2

LSTM_UNITS = 256
LSTM_MAX_STEP = 8

GAMMA = 0.99
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.0001
ALPHA = 1e-6  # the learning rate of optimizer
TAU = 0.001
UPDATE_FREQUENCY = 5  # the frequency to update target network

MAX_TIME_STEP = 10 * 10 ** 7
EPSILON_TIME_STEP = 1 * 10 ** 6  # for annealing the epsilon greedy
EPSILON_ANNEAL = float(INITIAL_EPSILON - FINAL_EPSILON) / EPSILON_TIME_STEP
BATCH_SIZE = 4
REPLAY_MEMORY = 2000

CHECKPOINT_DIR = 'tmp_drqn/checkpoints'
LOG_FILE = 'tmp_drqn/log'


class Network(object):

    def __init__(self, scope_name):

        with tf.variable_scope(scope_name) as scope:
            # input layer
            self.state_input = tf.placeholder('float', shape=[None, INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL])

            # hidden conv layer
            self.W_conv1 = weight_variable([8, 8, INPUT_CHANNEL, 32])
            self.b_conv1 = bias_variable([32])
            h_conv1 = tf.nn.relu(conv2d(self.state_input, self.W_conv1, 4) + self.b_conv1)

            h_poo1 = max_pool_2x2(h_conv1)

            self.W_conv2 = weight_variable([4, 4, 32, 64])
            self.b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_poo1, self.W_conv2, 2) + self.b_conv2)

            self.W_conv3 = weight_variable([3, 3, 64, 64])
            self.b_conv3 = bias_variable([64])
            h_conv3 = tf.nn.relu(conv2d(h_conv2, self.W_conv3, 1) + self.b_conv3)

            h_conv3_out_size = np.prod(h_conv3.get_shape().as_list()[1:])
            h_conv3_flat = tf.reshape(h_conv3, [-1, h_conv3_out_size])

            self.W_fc1 = weight_variable([h_conv3_out_size, LSTM_UNITS])
            self.b_fc1 = bias_variable([LSTM_UNITS])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, self.W_fc1) + self.b_fc1)

            # reshape to fit lstm (batch_size, timestep, LSTM_UNITS)
            self.timestep = tf.placeholder(dtype=tf.int32)
            self.batch_size = tf.placeholder(dtype=tf.int32)

            h_fc1_reshaped = tf.reshape(h_fc1, [self.batch_size, self.timestep, LSTM_UNITS])
            self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=LSTM_UNITS, state_is_tuple=True)
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

            # option1: for separate channel
            # streamA, streamV = tf.split(lstm_outputs, 2, axis=1)
            # self.AW = tf.Variable(tf.random_normal([LSTM_UNITS / 2, ACTIONS_DIM]))
            # self.VW = tf.Variable(tf.random_normal([LSTM_UNITS / 2, 1]))
            # advantage = tf.matmul(streamA, self.AW)
            # value = tf.matmul(streamV, self.VW)
            # self.Q_value = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keep_dims=True))

            # option2: for fully-connected
            self.W_fc2 = weight_variable([LSTM_UNITS, ACTIONS_DIM])
            self.b_fc2 = bias_variable([ACTIONS_DIM])
            self.Q_value = tf.matmul(lstm_outputs, self.W_fc2) + self.b_fc2

            self.Q_action = tf.argmax(self.Q_value, 1)
            print 'Q shape:', self.Q_value.get_shape()

            scope.reuse_variables()
            self.W_lstm = tf.get_variable("basic_lstm_cell/weights")
            self.b_lstm = tf.get_variable("basic_lstm_cell/biases")

        return

    def get_vars(self):
        return [
            self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_conv3, self.b_conv3,
            self.W_fc1, self.b_fc1,
            self.W_lstm, self.b_lstm,
            # self.AW, self.VW
            self.W_fc2, self.b_fc2,
        ]


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
        # update_target(self.session, self.target_ops)

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
        self.main_net = Network(scope_name='main')
        self.target_net = Network(scope_name='target')
        # self.target_ops = update_target_graph_op(tf.trainable_variables(), TAU)
        return

    def create_minimize(self):
        self.a = tf.placeholder('float', shape=[None, ACTIONS_DIM])
        self.y = tf.placeholder('float', shape=[None])
        Q_action = tf.reduce_sum(tf.multiply(self.main_net.Q_value, self.a), axis=1)
        self.full_loss = tf.reduce_mean(tf.square(self.y - Q_action))
        maskA = tf.zeros([BATCH_SIZE, LSTM_MAX_STEP // 2])
        maskB = tf.ones([BATCH_SIZE, LSTM_MAX_STEP // 2])
        mask = tf.concat([maskA, maskB], axis=1)
        mask = tf.reshape(mask, [-1])

        # just use a half loss with the mask:[0 0 0 0 1 1 1 1]
        self.loss = tf.multiply(self.full_loss, mask)

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=ALPHA)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=ALPHA, decay=0.99)
        self.gradients = tf.gradients(self.loss, self.main_net.get_vars())
        clip_grads = [tf.clip_by_norm(grad, 40.0) for grad in self.gradients]
        self.apply_gradients = self.optimizer.apply_gradients(zip(clip_grads, self.main_net.get_vars()))
        return

    def perceive(self, state, action, reward, next_state, terminal):
        self.global_t += 1

        self.episode_reward += reward
        if self.episode_start_time == 0.0:
            self.episode_start_time = time.time()

        if terminal or self.global_t % 20 == 0:
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

    def epsilon_greedy(self, state, lstm_state_in):
        """
        :param state: 1x84x84x3
        """
        Q_value_t, lstm_state_out = self.session.run(
            [self.main_net.Q_value, self.main_net.lstm_state],
            feed_dict={
                self.main_net.state_input: [state],
                self.main_net.initial_lstm_state: lstm_state_in,
                self.main_net.batch_size: 1,
                self.main_net.timestep: 1
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

        # if self.global_t % (UPDATE_FREQUENCY * 1000) == 0:
        #     update_target(self.session, self.target_ops)

        # limit the training frequency
        # if self.global_t % UPDATE_FREQUENCY != 0:
        #     return
        minibatch = self.replay_buffer.sample(BATCH_SIZE, LSTM_MAX_STEP)
        state_batch = [t[0] for t in minibatch]
        action_batch = [t[1] for t in minibatch]
        reward_batch = [t[2] for t in minibatch]
        next_state_batch = [t[3] for t in minibatch]
        terminal_batch = [t[4] for t in minibatch]

        y_batch = []
        # todo: need to feed with batch_size, timestep, lstm_state
        lstm_state_train = (np.zeros([BATCH_SIZE, LSTM_UNITS]), np.zeros([BATCH_SIZE, LSTM_UNITS]))
        Q_target = self.session.run(
            self.main_net.Q_value,
            feed_dict={
                self.main_net.state_input: next_state_batch,
                self.main_net.initial_lstm_state: lstm_state_train,
                self.main_net.batch_size: BATCH_SIZE,
                self.main_net.timestep: LSTM_MAX_STEP
            }
        )
        # Q_action = self.session.run(
        #     self.target_net.Q_action,
        #     feed_dict={
        #         self.target_net.state_input: next_state_batch,
        #         self.target_net.initial_lstm_state: lstm_state_train,
        #         self.target_net.batch_size: BATCH_SIZE,
        #         self.target_net.timestep: LSTM_MAX_STEP
        #     }
        # )
        for i in range(len(state_batch)):
            terminal = terminal_batch[i]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_target[i]))
                # y_batch.append(reward_batch[i] + GAMMA * Q_value[i][Q_action[i]])

        _, loss = self.session.run([self.apply_gradients, self.loss], feed_dict={
            self.y: y_batch,
            self.a: action_batch,
            self.main_net.state_input: state_batch,
            self.main_net.initial_lstm_state: lstm_state_train,
            self.main_net.batch_size: BATCH_SIZE,
            self.main_net.timestep: LSTM_MAX_STEP
        })

        # print loss
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
        self.summary_writer.flush()
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
        env.reset()
        episode_buffer = []
        lstm_state = (np.zeros([1, LSTM_UNITS]), np.zeros([1, LSTM_UNITS]))
        s_t = env.s_t
        while not env.terminal:
            # action_id = random.randint(0, 1)
            action_id, action_q, lstm_state = agent.epsilon_greedy(s_t, lstm_state)
            env.process(action_id)

            action = np.zeros(ACTIONS_DIM)
            action[action_id] = 1
            s_t1, reward, terminal = (env.s_t1, env.reward, env.terminal)
            # frame skip
            episode_buffer.append((s_t, action, reward, s_t1, terminal))
            agent.perceive(s_t, action, reward, s_t1, terminal)
            if agent.global_t % 10 == 0:
                print 'global_t:', agent.global_t, '/ epsilon:', agent.epsilon, '/ terminal:', terminal, \
                    '/ action:', action_id, '/ reward:', reward, '/ q_value:', action_q

            # s_t <- s_t1
            s_t = s_t1
            if len(episode_buffer) >= 50:
                # start a new episode buffer, in case of an over-long memory
                agent.replay_buffer.add(episode_buffer)
                episode_buffer = []
                print '----------- episode buffer > 100---------'
        # reset the state
        if len(episode_buffer) > LSTM_MAX_STEP:
            agent.replay_buffer.add(episode_buffer)
        print 'episode_buffer', len(episode_buffer)
        print 'replay_buffer.size:', agent.replay_buffer.size()
        # break
    return


if __name__ == '__main__':
    main()
