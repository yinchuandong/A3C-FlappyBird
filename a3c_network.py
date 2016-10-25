import tensorflow as tf
import numpy as np

from netutil import *


class A3CFFNetwork(object):

    def __init__(self, state_dim, state_chn, action_dim, device='/cpu:0'):
        self._state_dim = state_dim
        self._state_chn = state_chn
        self._action_dim = action_dim
        self._device = device
        self._create_network()
        return

    def _create_network(self):
        state_dim = self._state_dim
        state_chn = self._state_chn
        action_dim = self._action_dim
        with tf.device(self._device):
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

    def create_loss(self, entropy_beta):
        # taken action (input for policy)
        self.action_input = tf.placeholder('float', [None, self._action_dim])
        # temporary difference (R-V)  (input for policy)
        self.td = tf.placeholder('float', [None])

        # avoid NaN
        log_pi = tf.log(tf.clip_by_value(self.policy_output, 1e-20, 1.0))
        # policy entropy
        entropy = -tf.reduce_sum(self.policy_output * log_pi, reduction_indices=1)
        # policy loss L = log pi(a|s, theta) * (R - V)
        # (Adding minus, because the original paper's objective function is for gradient ascent,
        # but we use gradient descent optimizer.)
        policy_loss = -tf.reduce_sum(tf.reduce_sum(tf.mul(log_pi, self.action_input),
                                                   reduction_indices=1) * self.td + entropy * entropy_beta)

        # R (input for value)
        self.R = tf.placeholder('float', [None])
        # value loss (output) L = (R-V)^2
        value_loss = tf.reduce_mean(tf.square(self.R - self.value_output))
        self.total_loss = policy_loss + value_loss
        return

    def get_total_loss(self):
        return self.total_loss

    def get_vars(self):
        return [
            self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_conv3, self.b_conv3,
            self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3
        ]

    def sync_from(self, src_network, name=None):
        src_vars = src_network.get_vars()
        dst_vars = self.get_vars()

        sync_ops = []
        with tf.device(self._device):
            with tf.name_scope(name, 'A3CFFNetwork') as scope:
                for (src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_ops.append(tf.assign(dst_var, src_var))
                return tf.group(*sync_ops, name=scope)

    def run_policy_and_value(self, sess, state):
        policy, value = sess.run([self.policy_output, self.value_output], feed_dict={self.state_input: [state]})
        return policy[0], value[0]

    def run_policy(self, sess, state):
        policy = sess.run(self.policy_output, feed_dict={self.state_input: [state]})
        return policy[0]

    def run_value(self, sess, state):
        value = sess.run(self.value_output, feed_dict={self.state_input: [state]})
        return value[0]


if __name__ == '__main__':
    net = A3CFFNetwork(84, 3, 2)
    net.create_loss(0.01)
    print 'a3c_network.py'
