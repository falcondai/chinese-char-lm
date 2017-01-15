import numpy as np
import tensorflow as tf

def build_cnn(glyph_shape, code_dim, batch_size=None):
    glyph_ph = tf.placeholder('float', [batch_size] + list(glyph_shape))
    net = tf.expand_dims(glyph_ph / 255., -1)

    for i in xrange(4):
        net = tf.contrib.layers.convolution2d(
            inputs=net,
            num_outputs=32,
            kernel_size=(3, 3),
            stride=(2, 2),
            activation_fn=tf.nn.relu,
            biases_initializer=tf.zeros_initializer,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            scope='conv%i' % (i+1),
        )

    net = tf.contrib.layers.flatten(net)

    net = tf.contrib.layers.fully_connected(
        inputs=net,
        num_outputs=code_dim,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=tf.nn.relu,
        scope='fc1',
    )

    code = net

    return glyph_ph, code
