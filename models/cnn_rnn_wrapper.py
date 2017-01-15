import numpy as np
import tensorflow as tf

def build_lm(build_cnn, build_rnn, batch_size=None):
    inputs_ph = tf.placeholder('')
    seq_len_ph = tf.placeholder('int32', [batch_size])

    # define model or build a model from sub-models (in rnn/ or *_embed/)
    outputs = ...

    # return input placeholder and output tensors so callers can use them to build objective tensors
    return inputs_ph, seq_len_ph, outputs
