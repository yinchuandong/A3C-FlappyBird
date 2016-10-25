import os
import sys

import tensorflow as tf
import numpy as np
import math
import threading
import signal

from a3c_network import A3CFFNetwork
from a3c_actor_thread import A3CActorThread

from config import *


def log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1 - rate) + log_hi * rate
    return math.exp(v)


class A3C(object):

    def __init__(self):
        self.device = '/gpu:0' if USE_GPU else '/cpu:0'
        self.stop_requested = False
        self.global_t = 0
        self.global_network = A3CFFNetwork(STATE_DIM, STATE_CHN, ACTION_DIM, self.device)

        self.initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW, INITIAL_ALPHA_HIGH, INITIAL_ALPHA_LOG_RATE)
        self.learning_rate_input = tf.placeholder('float')
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate_input, RMSP_ALPHA, 0.0, 1e-10)

        self.actor_threads = []
        for i in range(PARALLEL_SIZE):
            actor_thread = A3CActorThread(i, self.global_network, self.initial_learning_rate,
                                          self.learning_rate_input, self.optimizer, MAX_TIME_STEP, self.device)
            self.actor_threads.append(actor_thread)

        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
        self.sess.run(tf.initialize_all_variables())

        self.saver = tf.train.Saver()
        self.restore()
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

    def train_function(self, parallel_index):
        actor_thread = self.actor_threads[parallel_index]
        while True:
            if self.stop_requested or (self.global_t > MAX_TIME_STEP):
                break
            diff_global_t = actor_thread.process(self.sess, self.global_t)
            self.global_t += diff_global_t

    def signal_handler(self, signal_, frame_):
        print 'You pressed Ctrl+C !'
        self.stop_requested = True
        return

    def run(self):
        train_treads = []
        for i in range(PARALLEL_SIZE):
            train_treads.append(threading.Thread(target=self.train_function, args=(i,)))

        signal.signal(signal.SIGINT, self.signal_handler)

        for t in train_treads:
            t.start()

        print 'Press Ctrl+C to stop'
        signal.pause()

        print 'Now saving data....'
        for t in train_treads:
            t.join()

        if not os.path.exists(CHECKPOINT_DIR):
            os.mkdir(CHECKPOINT_DIR)

        self.saver.save(self.sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step=self.global_t)
        return


if __name__ == '__main__':
    print 'a3c.py'
    net = A3C()
    net.run()
