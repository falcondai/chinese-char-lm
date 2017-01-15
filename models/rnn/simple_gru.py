import numpy as np
import tensorflow as tf

def build_rnn(code, batch_size=None):
    cell = tf.contrib.rnn.GRUBlockCell(128)
    seq_len_ph = tf.placeholder('int32', [batch_size])
    initial_state_ph = 

    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_input, seq_len_ph, initial_state_ph)
