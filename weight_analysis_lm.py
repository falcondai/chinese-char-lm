#!/usr/bin/env python

import numpy as np
# import cv2
import tensorflow as tf
# from tensorflow.contrib.tensorboard.plugins import projector
# from tensorflow.python import debug as tfdbg
import os, glob
from train_id_cnn_lm import render_glyph

def rebuild_model(token_ids, glyphs, seq_lens, vocab_size, embed_dim, rnn_dim, n_cnn_layers, n_cnn_filters):
    # encoder
    # encoder
    bs = tf.size(seq_lens)
    glyph_unaware = tf.contrib.layers.embed_sequence(token_ids, vocab_size, embed_dim)

    net = glyphs / 255.
    net = tf.reshape(net, (-1, 24, 24, 1))
    for i in xrange(n_cnn_layers):
        net = tf.contrib.layers.convolution2d(
            inputs=net,
            num_outputs=n_cnn_filters,
            kernel_size=(5, 5),
            stride=(2, 2),
            activation_fn=tf.nn.elu,
            biases_initializer=tf.zeros_initializer(),
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            scope='conv%i' % (i+1),
        )

    net = tf.contrib.layers.flatten(net)
    net = tf.contrib.layers.fully_connected(
        inputs=net,
        num_outputs=embed_dim,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,
        scope='embedding_fc',
    )
    
    glyph_aware = tf.reshape(net, (bs, -1, embed_dim))

    rnn_input = glyph_unaware + glyph_aware

    # rnn
    cell = tf.contrib.rnn.GRUBlockCell(rnn_dim)
    rnn_output, final_state = tf.nn.dynamic_rnn(cell, rnn_input, seq_lens, cell.zero_state(bs, 'float'))

    # decoder
    decoder_input = tf.reshape(rnn_output, (-1, rnn_dim))
    token_logit = tf.contrib.layers.fully_connected(
        inputs=decoder_input,
        num_outputs=2,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,
        scope='token_logit',
    )

    seq_logits = tf.reshape(token_logit, (bs, -1, 2))

    return glyph_unaware, glyph_aware

def compute_embeddings(checkpoint_dir, dict_path, vocab_size, n_oov_buckets, embed_dim):

        
    glyph_width = 24

    token_ids_ph = tf.placeholder('int32', shape=[None, None], name='center_token_id')

    # model
    glyph_ph = tf.placeholder('float', shape=[None, None, 24, 24], name='glyph')
    # token to token-id lookup

    seq_lens = tf.stack([tf.shape(token_ids_ph)[0]])

    # id_token_vocabulary = tf.contrib.lookup.index_to_string_table_from_file(dict_path, vocab_size=vocab_size)

    embed_dim, rnn_dim = 100, 64
    n_cnn_layers, n_cnn_filters = 1, 16
    seq_lens = [1]
    with tf.variable_scope('model'):
        id_emb, glyph_emb = rebuild_model(token_ids = token_ids_ph, glyphs = glyph_ph, seq_lens = seq_lens, vocab_size = vocab_size+n_oov_buckets, embed_dim = embed_dim, n_cnn_layers = n_cnn_layers, n_cnn_filters = n_cnn_filters, rnn_dim = rnn_dim)



    config = tf.ConfigProto(gpu_options={'allow_growth': True})
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        # saver = tf.train.import_meta_graph(checkpoint_dir)

        latest_checkpoint_path = tf.train.latest_checkpoint(os.path.dirname(checkpoint_dir))
        print '* restoring model from checkpoint %s' % latest_checkpoint_path
        saver.restore(sess, latest_checkpoint_path)

        # # Access the graph
        # graph = tf.get_default_graph()

        # # Retrieve variables
        # conv_bias = graph.get_tensor_by_name('model/conv1/biases:0')
        # conv_weight = graph.get_tensor_by_name('model/conv1/weights:0')
        # wemb_fc_weight = graph.get_tensor_by_name('model/embedding_fc/weights:0')
        # wemb_fc_bias = graph.get_tensor_by_name('model/embedding_fc/biases:0')
        # wemb_matrix = graph.get_tensor_by_name('model/EmbedSequence/embeddings:0')


        embeddings_val = np.zeros((vocab_size + n_oov_buckets, embed_dim))
        with open(dict_path, 'r') as fhandle:
            for i, line in enumerate(fhandle):
                if i > 4000:
                    break
                feed_dictionary = {token_ids_ph:[[i]], glyph_ph:[[render_glyph(line.strip().decode('utf8'))]]}

                id_emb_val, glyph_emb_val = sess.run([id_emb, glyph_emb], feed_dict=feed_dictionary)
                
        # embed_saver.save(sess, os.path.join(embeddings_checkpoint_dir, 'embeddings'))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--import-log-dir', required=True, help='path to skip-gram model checkpoint directory')
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-m', '--vocab-size', type=int, default=4000)
    parser.add_argument('-o', '--n-oov-buckets', type=int, default=1)
    parser.add_argument('--embed-dim', type=int, default=100)
    parser.add_argument('--dictionary', default='work/dict.txt', help='path to the dictionary file')

    args = parser.parse_args()

    compute_embeddings(args.import_log_dir, args.dictionary, args.vocab_size, args.n_oov_buckets, args.embed_dim)
