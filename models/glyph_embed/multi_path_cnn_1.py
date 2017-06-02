import numpy as np
import tensorflow as tf

def build_model(glyphs, embed_dim):
    net = glyphs / 255.

    # path 1 with square filters
    path1 = tf.contrib.layers.convolution2d(
      inputs=net,
      num_outputs=32,
      kernel_size=(7, 7),
      stride=(2, 2),
      activation_fn=tf.nn.relu,
      biases_initializer=tf.zeros_initializer(),
      weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
      scope='conv1.1',
    )

    path1 = tf.contrib.layers.convolution2d(
        inputs=path1,
        num_outputs=16,
        kernel_size=(5, 5),
        stride=(2, 2),
        activation_fn=tf.nn.relu,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        scope='conv1.2',
    )

    # path 2 with horizontal rectangular filter
    path2 = tf.contrib.layers.convolution2d(
        inputs=net,
        num_outputs=32,
        kernel_size=(9, 13),
        stride=(2, 2),
        activation_fn=tf.nn.relu,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        scope='conv2.1',
    )
    path2 = tf.contrib.layers.convolution2d(
        inputs=path2,
        num_outputs=16,
        kernel_size=(7, 11),
        stride=(2, 2),
        activation_fn=tf.nn.relu,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        scope='conv2.2',
    )

    # path 3 with vertical rectangular filter
    path3 = tf.contrib.layers.convolution2d(
        inputs=net,
        num_outputs=32,
        kernel_size=(13, 9),
        stride=(2, 2),
        activation_fn=tf.nn.relu,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        scope='conv3.1',
    )
    path3 = tf.contrib.layers.convolution2d(
        inputs=path3,
        num_outputs=16,
        kernel_size=(11, 7),
        stride=(2, 2),
        activation_fn=tf.nn.relu,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        scope='conv3.2',
    )

    net = tf.contrib.layers.flatten(path1 + path2 + path3)
    net = tf.contrib.layers.fully_connected(
      inputs=net,
      num_outputs=embed_dim,
      biases_initializer=tf.zeros_initializer(),
      weights_initializer=tf.contrib.layers.xavier_initializer(),
      activation_fn=None,
      scope='embedding_fc',
    )

    return net
