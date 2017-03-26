#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
# import cv2
import tensorflow as tf
import os, argparse, importlib

from render import render_text, ascii_print

# from models.cnn.conv4 import build_cnn
# from models.rnn.simple_gru import build_rnn

def conform(im, shape):
    h, w = shape
    ih, iw = im.shape
    if ih == h:
        return im
    if ih > h:
        # take the center crop
        y0 = (ih - h) / 2
        return im[y0:y0 + h]
    # shorter image, ih < h
    new_im = np.zeros(shape)
    y0 = (h - ih) / 2
    new_im[y0:y0 + ih] = im
    return new_im

def render_glyph(char, shape=(24, 24), font=None):
    return conform(render_text(char, font), shape)
    # return cv2.resize(render_text(char, font), shape)

def build_model(token_ids, seq_lens, vocab_size, embed_dim, rnn_dim):
    # encoder
    rnn_input = tf.contrib.layers.embed_sequence(token_ids, vocab_size, embed_dim)

    # rnn
    bs = tf.size(seq_lens)
    cell = tf.contrib.rnn.GRUBlockCell(rnn_dim)
    rnn_output, final_state = tf.nn.dynamic_rnn(cell, rnn_input, seq_lens, cell.zero_state(bs, 'float'))

    # decoder
    decoder_input = tf.reshape(rnn_output, (-1, rnn_dim))
    token_logit = tf.contrib.layers.fully_connected(
        inputs=decoder_input,
        num_outputs=vocab_size,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,
        scope='token_logit',
    )

    seq_logits = tf.reshape(token_logit, (bs, -1, vocab_size))

    return seq_logits, final_state

def train():
    vocab_size = 200
    n_oov_buckets = 1
    batch_size = 4

    # input pipeline
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once('work/train*.txt'), shuffle=True)
    reader = tf.TextLineReader()
    _, line = reader.read(filename_queue)
    seq_len = tf.shape(tf.string_split([line]).values)[0]

    batch = tf.train.shuffle_batch([line, seq_len], batch_size=batch_size, capacity=128, min_after_dequeue=32)

    # token to token-id lookup
    vocabulary = tf.contrib.lookup.string_to_index_table_from_file('work/dict.txt', num_oov_buckets=n_oov_buckets, vocab_size=vocab_size)

    tokens = tf.string_split(batch[0])
    ids = tf.sparse_tensor_to_dense(vocabulary.lookup(tokens), validate_indices=False)
    seq_lens = batch[1]

    # model
    seq_logits, final_state = build_model(ids, seq_lens, vocab_size + n_oov_buckets, 100, 64)

    # loss
    mask = tf.sequence_mask(seq_lens, dtype=tf.float32)
    loss = tf.contrib.seq2seq.sequence_loss(seq_logits, ids, mask, average_across_timesteps=True, average_across_batch=True)
    n_samples = tf.reduce_sum(mask)

    update_op = tf.train.AdamOptimizer().minimize(loss * n_samples)

    with tf.Session() as sess:
        tf.tables_initializer().run()
        tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        while True:
            loss_val, _ = sess.run([loss, update_op])
            print loss_val
            # token_val, id_val, seq_len_val = sess.run([tokens, ids, seq_lens])
            # print token_val
            # print id_val, seq_len_val

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    # s = u'你好吗？'
    # for c in s:
    #     a = render_glyph(c)
    #     print a.shape
    #     # print a
    #     # ascii_print(a)
    train()
