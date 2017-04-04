import tensorflow as tf
import numpy as np
import random
import time
import os
import sys
from netutil import *
from game.flappy_bird import FlappyBird
from collections import deque


INPUT_SIZE = 84
INPUT_CHANNEL = 4
ACTIONS_DIM = 2

LSTM_UNITS = 512

GAMMA = 0.99
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.0001
ALPHA = 1e-6  # the learning rate of optimizer

MAX_TIME_STEP = 10 * 10 ** 7
EPSILON_TIME_STEP = 1 * 10 ** 6  # for annealing the epsilon greedy
EPSILON_ANNEAL = float(INITIAL_EPSILON - FINAL_EPSILON) / EPSILON_TIME_STEP
BATCH_SIZE = 32
REPLAY_MEMORY = 20000

CHECKPOINT_DIR = 'tmp_dqn/checkpoints'
LOG_FILE = 'tmp_dqn/log'


class Network(object):

    def __init__(self, scope_name):

        with tf.variable_scope(scope_name) as scope:
            # input layer
            self.state_input = tf.placeholder('float', shape=[None, INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL])

            # hidden conv layer
            self.W_conv1 = weight_variable([8, 8, INPUT_CHANNEL, 32])
            self.b_conv1 = bias_variable([32])
            h_conv1 = tf.nn.relu(conv2d(self.state_input, self.W_conv1, 4) + self.b_conv1)

            self.W_conv2 = weight_variable([4, 4, 32, 64])
            self.b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

            self.W_conv3 = weight_variable([3, 3, 64, 64])
            self.b_conv3 = bias_variable([64])
            h_conv3 = tf.nn.relu(conv2d(h_conv2, self.W_conv3, 1) + self.b_conv3)

            h_conv3_out_size = np.prod(h_conv3.get_shape().as_list()[1:])
            h_conv3_flat = tf.reshape(h_conv3, [-1, h_conv3_out_size])

            self.W_fc1 = weight_variable([h_conv3_out_size, LSTM_UNITS])
            self.b_fc1 = bias_variable([LSTM_UNITS])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, self.W_fc1) + self.b_fc1)

            self.W_fc2 = weight_variable([LSTM_UNITS, ACTIONS_DIM])
            self.b_fc2 = bias_variable([ACTIONS_DIM])
            self.Q_value = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2

        return

    def get_vars(self):
        return [
            self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2,
        ]


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
        self.main_net = Network(scope_name='main')
        self.target_net = Network(scope_name='target')
        return

    def create_minimize(self):
        self.a = tf.placeholder('float', shape=[None, ACTIONS_DIM])
        self.y = tf.placeholder('float', shape=[None])
        Q_action = tf.reduce_sum(tf.multiply(self.main_net.Q_value, self.a), axis=1)
        self.loss = tf.reduce_mean(tf.square(self.y - Q_action))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=ALPHA)
        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=ALPHA, decay=0.99)
        self.apply_gradients = self.optimizer.minimize(self.loss)
        # self.gradients = tf.gradients(self.loss, self.main_net.get_vars())
        # clip_grads = [tf.clip_by_norm(grad, 40.0) for grad in self.gradients]
        # self.apply_gradients = self.optimizer.apply_gradients(zip(clip_grads, self.main_net.get_vars()))
        return

    def perceive(self, state, action, reward, next_state, terminal):
        self.global_t += 1

        self.replay_buffer.append((state, action, reward, next_state, terminal))

        self.episode_reward += reward
        if self.episode_start_time == 0.0:
            self.episode_start_time = time.time()

        if terminal or self.global_t % 20 == 0:
            living_time = time.time() - self.episode_start_time
            self.record_log(self.episode_reward, living_time)

        if terminal:
            self.episode_reward = 0.0
            self.episode_start_time = time.time()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

        if self.global_t % 100000 == 0:
            self.backup()
        return

    def epsilon_greedy(self, state):
        """
        :param state: 1x84x84x3
        """
        Q_value_t = self.session.run(
            self.main_net.Q_value,
            feed_dict={
                self.main_net.state_input: [state],
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
        return action_index, max_q_value

    def train_Q_network(self):
        '''
        do backpropogation
        '''
        # len(minibatch) = BATCH_SIZE * LSTM_MAX_STEP

        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [t[0] for t in minibatch]
        action_batch = [t[1] for t in minibatch]
        reward_batch = [t[2] for t in minibatch]
        next_state_batch = [t[3] for t in minibatch]
        terminal_batch = [t[4] for t in minibatch]

        y_batch = []
        Q_target = self.session.run(
            self.main_net.Q_value,
            feed_dict={
                self.main_net.state_input: next_state_batch,
            }
        )

        for i in range(len(minibatch)):
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
    agent = DQN()
    env = FlappyBird()
    env.reset()

    while True:
        action_id, action_q = agent.epsilon_greedy(env.s_t)
        env.process(action_id)

        action = np.zeros(ACTIONS_DIM)
        action[action_id] = 1
        state = env.s_t
        next_state = env.s_t1
        reward = env.reward
        terminal = env.terminal

        agent.perceive(state, action, reward, next_state, terminal)

        if agent.global_t % 10:
            print 'global_t:', agent.global_t, '/ epsilon:', agent.epsilon, '/ terminal:', terminal, \
                '/ action:', action_id, '/ reward:', reward, '/ q_value:', action_q

        if terminal:
            env.reset()
        env.update()
        # break
    return


if __name__ == '__main__':
    main()
