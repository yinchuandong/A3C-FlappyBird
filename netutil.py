import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def output_size(in_size, filter_size, stride):
    return (in_size - filter_size) / stride + 1


def lstm_last_relevant(output, length):
    '''
    get the last relevant frame of the output of tf.nn.dynamica_rnn()
    '''
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, index)
    return relevant


def update_target_graph_op(trainable_vars, tau=0.001):
    '''
    theta_prime = tau * theta + (1 - tau) * theta_prime
    '''
    size = len(trainable_vars)
    update_ops = []
    for i, var in enumerate(trainable_vars[0:size / 2]):
        target = trainable_vars[size // 2 + i]
        op = tf.assign(target, tau * var.value() + (1 - tau) * target.value())
        update_ops.append(op)
    return update_ops


def update_target(session, update_ops):
    session.run(update_ops)
    tf_vars = tf.trainable_variables()
    size = len(tf.trainable_variables())
    theta = session.run(tf_vars[0])
    theta_prime = session.run(tf_vars[size // 2])
    assert(theta.all() == theta_prime.all())
    return
