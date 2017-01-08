import tensorflow as tf
import numpy as np
import random
import time
from netutil import *
from game.game_state import GameState
from replay_buffer import ReplayBuffer

INPUT_SIZE = 84
INPUT_CHANNEL = 4
ACTIONS_DIM = 2

LSTM_UNITS = 256

GAMMA = 0.99
FINAL_EPSILON = 0.01
INITIAL_EPSILON = 1.0

ALPHA = 1e-6  # the learning rate of optimizer

MAX_TIME_STEP = 10 * 10 ** 7
EPSILON_TIME_STEP = 1 * 10 ** 6  # for annealing the epsilon greedy
REPLAY_MEMORY = 50000
BATCH_SIZE = 32

CHECKPOINT_DIR = 'tmp-dqn/checkpoints'
LOG_FILE = 'tmp-dqn/log'


class DQN(object):

    def __init__(self):
        self.global_t = 0
        self.replay_buffer = ReplayBuffer(REPLAY_MEMORY)
        self.epsilon = INITIAL_EPSILON

        # q-network parameter
        self.create_network()
        self.create_minimize()

        # init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

        self.saver = tf.train.Saver(tf.all_variables())
        self.restore()

        # for recording the log into tensorboard
        self.time_input = tf.placeholder(tf.float32)
        self.reward_input = tf.placeholder(tf.float32)
        tf.scalar_summary('living_time', self.time_input)
        tf.scalar_summary('reward', self.reward_input)
        self.summary_op = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter(LOG_FILE, self.sess.graph)

        self.episode_start_time = 0.0
        self.episode_reward = 0.0
        return

    def create_network(self):
        # input layer
        s = tf.placeholder('float', shape=[None, INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL], name='s')

        # hidden conv layer
        W_conv1 = weight_variable([8, 8, INPUT_CHANNEL, 32], name='W_conv1')
        b_conv1 = bias_variable([32], name='b_conv1')
        h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)

        W_conv2 = weight_variable([4, 4, 32, 64], name='W_conv2')
        b_conv2 = bias_variable([64], name='b_conv2')
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

        W_conv3 = weight_variable([3, 3, 64, 64], name='W_conv3')
        b_conv3 = bias_variable([64], name='b_conv3')
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

        h_conv3_out_size = np.prod(h_conv3.get_shape().as_list()[1:])
        print h_conv3_out_size
        h_conv3_flat = tf.reshape(h_conv3, [-1, h_conv3_out_size], name='h_conv3_flat')

        W_fc1 = weight_variable([h_conv3_out_size, 256])
        b_fc1 = bias_variable([256])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # reshape to fit lstm (1, 5, 256)
        h_fc1_reshaped = tf.reshape(h_fc1, [BATCH_SIZE, -1, 256])

        self.lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=LSTM_UNITS, state_is_tuple=True)
        self.lstm_initial_state = self.lstm_cell.zero_state(BATCH_SIZE)
        self.timestep = tf.placeholder(dtype=tf.int32)
        lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(
            self.lstm_cell,
            h_fc1_reshaped,
            initial_state=self.initial_lstm_state,
            sequence_length=self.timestep,
            time_major=False,
            scope=scope
        )

        # readout layer: Q_value
        W_fc2 = weight_variable([512, ACTIONS_DIM], name='W_fc2')
        b_fc2 = bias_variable([ACTIONS_DIM], name='b_fc2')
        Q_value = tf.matmul(h_fc1, W_fc2) + b_fc2

        self.s = s
        self.Q_value = Q_value
        return

    def create_minimize(self):
        self.a = tf.placeholder('float', shape=[None, ACTIONS_DIM], name='a')
        self.y = tf.placeholder('float', shape=[None], name='y')
        Q_action = tf.reduce_sum(tf.mul(self.Q_value, self.a), reduction_indices=1)
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

        if len(self.replay_buffer) > REPLAY_MEMORY:
            self.train_Q_network()
        return

    def get_action_index(self, state):
        Q_value_t = self.session.run(self.Q_value, feed_dict={self.s: state})[0]
        return np.argmax(Q_value_t), np.max(Q_value_t)

    def epsilon_greedy(self, state):
        """
        :param state: 1x84x84x3
        """
        Q_value_t = self.Q_value.eval(session=self.session, feed_dict={self.s: state})[0]
        action_index = 0
        if random.random() <= self.epsilon:
            action_index = random.randrange(ACTIONS_DIM)
        else:
            action_index = np.argmax(Q_value_t)

        if self.epsilon > FINAL_EPSILON and self.global_t > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / OBSERVE
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
        Q_value_batch = self.session.run(Q_value, feed_dict={self.s: next_state_batch})
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
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
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

        self.saver.save(self.sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step=self.global_t)
        return


def run_doom():
    '''
    the function for training
    '''
    agent = DQN()
    game = GameState()
    game.reset()

    while agent.global_t < MAX_TIME_STEP:
        action_id = agent.epsilon_greedy(game.s_t)
        game.process(action_id)
        action = np.zeros(ACTIONS_DIM)
        action[action_id] = 1
        agent.perceive(game.s_t, action, game.reward, game.s_t1, game.terminal)

        if game.terminal:
            game.reset()
        # s_t <- s_t1
        game.update()

    return


if __name__ == '__main__':
    run_doom()
