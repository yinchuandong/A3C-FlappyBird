import tensorflow as tf
import numpy as np


class AccumTrainer(object):

    def __init__(self, device='/cpu:0', name='Trainer'):
        self._device = device
        self._name = name
        return

    def _create_accum_grad(self, var):
        # Create Variable where to accumulate gradients
        # It has the same shape as inputted var
        zero = tf.zeros(var.get_shape().as_list(), dtype=var.dtype)
        name = var.name.replace(":", "_") + "_accum_grad"
        accum_grad = tf.Variable(zero, name=name, trainable=False)
        return accum_grad

    def create_minimize(self, loss, var_list):
        with tf.device(self._device):
            # todo: check the necessarity of var_refs
            var_refs = [v.ref() for v in var_list]
            self._var_list = var_list
            self._grad_list = tf.gradients(loss, var_refs)
            self._accum_grad_list = []

            with tf.control_dependencies(None):
                for var in var_list:
                    accum_grad = self._create_accum_grad(var)
                    self._accum_grad_list.append(accum_grad)
        return

    def get_accum_grad_list(self):
        return self._accum_grad_list

    def accumulate_gradients(self, name=None):
        with tf.device(self._device):
            accum_ops = []
            with tf.name_scope(name, self._name) as scope:
                for var, grad, accum_grad in zip(self._var_list, self._grad_list, self._accum_grad_list):
                    with tf.name_scope('accum_' + var.op.name):
                        accum_ops.append(tf.assign_add(accum_grad, grad))
                return tf.group(*accum_ops, name=scope)

    def reset_gradients(self, name=None):
        with tf.device(self._device):
            reset_ops = []
            with tf.name_scope(name, self._name) as scope:
                for var, accum_grad in zip(self._var_list, self._accum_grad_list):
                    with tf.name_scope('reset_' + var.op.name):
                        zero = tf.zeros(var.get_shape())
                        reset = accum_grad.assign(zero)
                        reset_ops.append(reset)
                return tf.group(*reset_ops, name=scope)


if __name__ == '__main__':
    print 'he'
