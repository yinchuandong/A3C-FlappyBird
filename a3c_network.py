import tensorflow as tf
import numpy as np

from netutil import *
from config import LSTM_UNITS


class A3CNetwork(object):

    def __init__(self, state_dim, state_chn, action_dim, device='/cpu:0'):
        self._state_dim = state_dim
        self._state_chn = state_chn
        self._action_dim = action_dim
        self._device = device
        return

    def create_loss(self, entropy_beta):
        # taken action (input for policy)
        self.action_input = tf.placeholder(tf.float32, [None, self._action_dim], name="actions")
        # R (input for value)
        self.R = tf.placeholder(tf.float32, [None], name="R")
        # temporary difference (R-V)  (input for policy)
        self.td = tf.placeholder(tf.float32, [None], name="td")

        # avoid NaN
        log_pi = tf.log(tf.clip_by_value(self.policy_output, 1e-20, 1.0))
        # policy entropy
        entropy = -tf.reduce_sum(self.policy_output * log_pi, axis=1)
        # policy loss L = log pi(a|s, theta) * (R - V)
        # (Adding minus, because the original paper's objective function is for gradient ascent,
        # but we use gradient descent optimizer.)
        policy_loss = -tf.reduce_sum(tf.reduce_sum(tf.multiply(log_pi, self.action_input),
                                                   axis=1) * self.td + entropy * entropy_beta)

        # value loss (output) L = (R-V)^2
        # value_loss = tf.reduce_mean(tf.square(self.R - self.value_output))
        value_loss = 0.5 * tf.nn.l2_loss(self.R - self.value_output)
        # value_loss = 0.5 * tf.nn.l2_loss(self.td)

        self.total_loss = policy_loss + value_loss
        return

    def run_policy_and_value(self, sess, state):
        raise NotImplementedError()

    def run_policy(self, sess, state):
        raise NotImplementedError()

    def run_value(self, sess, state):
        raise NotImplementedError()

    def get_vars(self):
        raise NotImplementedError()

    def sync_from(self, src_network, name=None):
        src_vars = src_network.get_vars()
        dst_vars = self.get_vars()

        sync_ops = []
        with tf.device(self._device):
            with tf.name_scope(name, 'A3CNetwork') as scope:
                for (src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_ops.append(tf.assign(dst_var, src_var))
                return tf.group(*sync_ops, name=scope)


class A3CFFNetwork(A3CNetwork):

    def __init__(self, state_dim, state_chn, action_dim, device='/cpu:0', thread_index='-1'):
        A3CNetwork.__init__(self, state_dim, state_chn, action_dim, device)
        self._thread_index = thread_index
        self._create_network()
        return

    def _create_network(self):
        state_dim = self._state_dim
        state_chn = self._state_chn
        action_dim = self._action_dim
        scope_name = "net_" + str(self._thread_index)
        with tf.device(self._device), tf.variable_scope(scope_name) as scope:
            # state input
            self.state_input = tf.placeholder('float', [None, state_dim, state_dim, state_chn])

            # conv1
            self.W_conv1 = weight_variable([8, 8, state_chn, 16])
            self.b_conv1 = bias_variable([16])
            h_conv1 = tf.nn.relu(conv2d(self.state_input, self.W_conv1, 4) + self.b_conv1)

            # conv2
            self.W_conv2 = weight_variable([4, 4, 16, 32])
            self.b_conv2 = bias_variable([32])
            h_conv2 = tf.nn.relu(conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

            # conv3
            self.W_conv3 = weight_variable([3, 3, 32, 64])
            self.b_conv3 = bias_variable([64])
            h_conv3 = tf.nn.relu(conv2d(h_conv2, self.W_conv3, 1) + self.b_conv3)

            h_conv3_out_size = np.prod(h_conv3.get_shape().as_list()[1:])
            print 'h_conv3_out_size', h_conv3_out_size
            h_conv3_flat = tf.reshape(h_conv3, [-1, h_conv3_out_size])

            # fc1
            self.W_fc1 = weight_variable([h_conv3_out_size, 512])
            self.b_fc1 = bias_variable([512])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, self.W_fc1) + self.b_fc1)

            # fc2: (pi) for policy output
            self.W_fc2 = weight_variable([512, action_dim])
            self.b_fc2 = bias_variable([action_dim])
            self.policy_output = tf.nn.softmax(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)

            # fc3: (v)  for value output
            self.W_fc3 = weight_variable([512, 1])
            self.b_fc3 = bias_variable([1])
            v_ = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3
            self.value_output = tf.reshape(v_, [-1])
        return

    def get_vars(self):
        return [
            self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_conv3, self.b_conv3,
            self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3
        ]

    def run_policy_and_value(self, sess, state):
        policy, value = sess.run([self.policy_output, self.value_output], feed_dict={self.state_input: [state]})
        return policy[0], value[0]

    def run_policy(self, sess, state):
        policy = sess.run(self.policy_output, feed_dict={self.state_input: [state]})
        return policy[0]

    def run_value(self, sess, state):
        value = sess.run(self.value_output, feed_dict={self.state_input: [state]})
        return value[0]


class A3CLSTMNetwork(A3CNetwork):
    def __init__(self, state_dim, state_chn, action_dim, device='/cpu:0', thread_index=-1):
        '''
        Args:
            thread_index: int, -1 means global network
        '''
        A3CNetwork.__init__(self, state_dim, state_chn, action_dim, device)
        self._thread_index = thread_index
        self._create_network()
        return

    def _create_network(self):
        state_dim = self._state_dim
        state_chn = self._state_chn
        action_dim = self._action_dim
        scope_name = 'net_' + str(self._thread_index)
        with tf.device(self._device), tf.variable_scope(scope_name) as scope:
            # state input
            self.state_input = tf.placeholder('float', [None, state_dim, state_dim, state_chn])
            self.state_input = self.state_input / 255.0

            # conv1
            self.W_conv1 = weight_variable([8, 8, state_chn, 16])
            self.b_conv1 = bias_variable([16])
            h_conv1 = tf.nn.relu(conv2d(self.state_input, self.W_conv1, 4) + self.b_conv1)

            # conv2
            self.W_conv2 = weight_variable([4, 4, 16, 32])
            self.b_conv2 = bias_variable([32])
            h_conv2 = tf.nn.relu(conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

            h_conv2_out_size = np.prod(h_conv2.get_shape().as_list()[1:])
            print 'h_conv2_out_size', h_conv2_out_size
            h_conv2_flat = tf.reshape(h_conv2, [-1, h_conv2_out_size])

            # conv3
            # self.W_conv3 = weight_variable([3, 3, 32, 64])
            # self.b_conv3 = bias_variable([64])
            # h_conv3 = tf.nn.relu(conv2d(h_conv2, self.W_conv3, 1) + self.b_conv3)

            # h_conv3_out_size = np.prod(h_conv3.get_shape().as_list()[1:])
            # print 'h_conv3_out_size', h_conv3_out_size
            # h_conv3_flat = tf.reshape(h_conv3, [-1, h_conv3_out_size])

            # fc1
            self.W_fc1 = weight_variable([h_conv2_out_size, 256])
            self.b_fc1 = bias_variable([256])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)

            # reshape to fit lstm (1, 5, 256)
            h_fc1_reshaped = tf.reshape(h_fc1, [1, -1, LSTM_UNITS])

            self.step_size = tf.placeholder('float', [1])
            self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=LSTM_UNITS, state_is_tuple=True)
            self.initial_lstm_state = self.lstm_cell.zero_state(1, tf.float32)

            lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(
                self.lstm_cell,
                h_fc1_reshaped,
                initial_state=self.initial_lstm_state,
                sequence_length=self.step_size,
                time_major=False,
                scope=scope
            )
            print lstm_outputs.get_shape()
            lstm_outputs = tf.reshape(lstm_outputs, [-1, LSTM_UNITS])

            # fc2: (pi) for policy output
            self.W_fc2 = weight_variable([256, action_dim])
            self.b_fc2 = bias_variable([action_dim])
            self.policy_output = tf.nn.softmax(tf.matmul(lstm_outputs, self.W_fc2) + self.b_fc2)

            # fc3: (v)  for value output
            self.W_fc3 = weight_variable([256, 1])
            self.b_fc3 = bias_variable([1])
            v_ = tf.matmul(lstm_outputs, self.W_fc3) + self.b_fc3
            self.value_output = tf.reshape(v_, [-1])

            scope.reuse_variables()
            self.W_lstm = tf.get_variable("basic_lstm_cell/weights")
            self.b_lstm = tf.get_variable("basic_lstm_cell/biases")

            self.reset_lstm_state()
        return

    def reset_lstm_state(self):
        self.lstm_state_out = (np.zeros([1, LSTM_UNITS]), np.zeros([1, LSTM_UNITS]))
        return

    def get_vars(self):
        return [
            self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            # self.W_conv3, self.b_conv3,
            self.W_fc1, self.b_fc1,
            self.W_lstm, self.b_lstm,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3
        ]

    def run_policy_and_value(self, sess, state):
        policy, value, self.lstm_state_out = sess.run(
            [self.policy_output, self.value_output, self.lstm_state],
            feed_dict={
                self.state_input: [state],
                self.initial_lstm_state: self.lstm_state_out,
                self.step_size: [1]
            }
        )
        return policy[0], value[0]

    def run_policy(self, sess, state):
        policy, self.lstm_state_out = sess.run(
            [self.policy_output, self.lstm_state],
            feed_dict={
                self.state_input: [state],
                self.initial_lstm_state: self.lstm_state_out,
                self.step_size: [1]
            }
        )
        return policy[0]

    def run_value(self, sess, state):
        # don't need to change state
        prev_lstm_state_out = self.lstm_state_out
        value, _ = sess.run(
            [self.value_output, self.lstm_state],
            feed_dict={
                self.state_input: [state],
                self.initial_lstm_state: self.lstm_state_out,
                self.step_size: [1]
            }
        )
        self.lstm_state_out = prev_lstm_state_out
        return value[0]


if __name__ == '__main__':
    # net = A3CFFNetwork(84, 3, 2)
    # net.create_loss(0.01)
    net = A3CLSTMNetwork(84, 3, 2)
    net.create_loss(0.01)
    print 'a3c_network.py'
