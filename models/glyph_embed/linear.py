import numpy as np
import tensorflow as tf

def build_model(glyphs, embed_dim):
    net = glyphs / 255.

    net = tf.contrib.layers.flatten(net)
    net = tf.contrib.layers.fully_connected(
      inputs=net,
      num_outputs=embed_dim,
      biases_initializer=tf.zeros_initializer(),
      weights_initializer=tf.contrib.layers.xavier_initializer(),
      activation_fn=None,
      scope='embedding_fc',
    )

    return net
