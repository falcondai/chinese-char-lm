#!/usr/bin/env python

import numpy as np
# import cv2
import tensorflow as tf
# from tensorflow.contrib.tensorboard.plugins import projector
# from tensorflow.python import debug as tfdbg
import os, glob
from train_id_cnn_lm import render_glyph, build_model
from models.glyph_embed import multi_path_cnn_1
from train_id_cnn_Biseg_cpu import build_model as biseg_build_model
from models.glyph_embed import simple_cnn_2

def rebuild_simple_model(token_ids, glyphs, seq_lens, vocab_size, embed_dim, rnn_dim, n_cnn_layers, n_cnn_filters, n_oov_buckets):
    # encoder
    bs = tf.size(seq_lens)
    glyph_unaware = tf.contrib.layers.embed_sequence(token_ids, vocab_size + n_oov_buckets, embed_dim)

    # glyph-aware
    glyphs = tf.reshape(glyphs, (-1, 24, 24, 1))
    # linear glyph embedder
    net = simple_cnn_2.build_model(glyphs, embed_dim)

    glyph_aware = tf.reshape(net, (bs, -1, embed_dim))

    in_vocab = tf.expand_dims(tf.cast(tf.less(token_ids, vocab_size), 'float'), -1)
    # msr-m1, msr-m0, msr-l0
    # rnn_input = glyph_unaware + in_vocab * glyph_aware
    # msr-i0
    # rnn_input = glyph_unaware + 0. * glyph_aware
    # msr-l1, msr-c2
    rnn_input = 0. * glyph_unaware + glyph_aware

    # rnn
    cell = tf.contrib.rnn.GRUBlockCell(rnn_dim)
    rnn_output, final_state = tf.nn.dynamic_rnn(cell, rnn_input, seq_lens, cell.zero_state(bs, 'float'))

    # decoder
    decoder_input = tf.reshape(rnn_output, (-1, rnn_dim))
    token_logit = tf.contrib.layers.fully_connected(
        inputs=decoder_input,
        num_outputs=vocab_size + n_oov_buckets,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,
        scope='token_logit',
    )

    seq_logits = tf.reshape(token_logit, (bs, -1, vocab_size + n_oov_buckets))

    return seq_logits, final_state, glyph_unaware, glyph_aware

def rebuild_multi_path_model(token_ids, glyphs, seq_lens, vocab_size, embed_dim, rnn_dim, n_oov_buckets):
    bs = tf.size(seq_lens)
    glyph_unaware = tf.contrib.layers.embed_sequence(token_ids, vocab_size + n_oov_buckets, embed_dim)


    # glyph-aware
    glyphs = tf.reshape(glyphs, (-1, 24, 24, 1))
    # linear glyph embedder
    net = multi_path_cnn_1.build_model(glyphs, embed_dim)

    glyph_aware = tf.reshape(net, (bs, -1, embed_dim))

    in_vocab = tf.expand_dims(tf.cast(tf.less(token_ids, vocab_size), 'float'), -1)
    # msr-m1, msr-m0, msr-l0
    # rnn_input = glyph_unaware + in_vocab * glyph_aware
    # msr-i0
    # rnn_input = glyph_unaware + 0. * glyph_aware
    # msr-l1, msr-c2
    rnn_input = 0. * glyph_unaware + glyph_aware

    # rnn
    cell = tf.contrib.rnn.GRUBlockCell(rnn_dim)
    rnn_output, final_state = tf.nn.dynamic_rnn(cell, rnn_input, seq_lens, cell.zero_state(bs, 'float'))

    # decoder
    decoder_input = tf.reshape(rnn_output, (-1, rnn_dim))
    token_logit = tf.contrib.layers.fully_connected(
        inputs=decoder_input,
        num_outputs=vocab_size + n_oov_buckets,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,
        scope='token_logit',
    )

    seq_logits = tf.reshape(token_logit, (bs, -1, vocab_size + n_oov_buckets))

    return glyph_unaware, glyph_aware

def compute_embeddings(checkpoint_dir, dict_path, vocab_size, n_oov_buckets, embed_dim, rnn_dim, task):

    id_emb_accu = 0.0
    glyph_emb_accu = 0.0
    id_glyph_ratio_acc = []

    glyph_width = 24

    token_ids_ph = tf.placeholder('int32', shape=[None, None], name='center_token_id')

    # model
    glyph_ph = tf.placeholder('float', shape=[None, None, 24, 24], name='glyph')
    # token to token-id lookup

    seq_lens = tf.stack([tf.shape(token_ids_ph)[0]])

    # id_token_vocabulary = tf.contrib.lookup.index_to_string_table_from_file(dict_path, vocab_size=vocab_size)

    embed_dim, rnn_dim = embed_dim, rnn_dim
    seq_lens = [1]
    with tf.variable_scope('model'):
        if task == 'lm':
            n_cnn_layers, n_cnn_filters = 1, 16

            _, _, id_emb, glyph_emb = build_model(token_ids = token_ids_ph, glyphs = glyph_ph, seq_lens = seq_lens, vocab_size = vocab_size + n_oov_buckets, embed_dim = embed_dim, rnn_dim = rnn_dim, n_cnn_layers = n_cnn_layers, n_cnn_filters = n_cnn_filters)
        elif task == 'lm_multi':
            id_emb, glyph_emb = rebuild_multi_path_model(token_ids = token_ids_ph, glyphs = glyph_ph, seq_lens = seq_lens, vocab_size = vocab_size, embed_dim = embed_dim, rnn_dim = rnn_dim, n_oov_buckets = n_oov_buckets)
        elif task == 'biseg':
            n_cnn_layers, n_cnn_filters = 1, 16 
            _, _, id_emb, glyph_emb = biseg_build_model(token_ids = token_ids_ph, glyphs= glyph_ph, seq_lens = seq_lens, vocab_size = vocab_size + n_oov_buckets, embed_dim = embed_dim, rnn_dim = rnn_dim, n_cnn_layers = n_cnn_layers, n_cnn_filters = n_cnn_filters)
        elif task == 'lm_simple':
            n_cnn_layers, n_cnn_filters = 1, 16

            _, _, id_emb, glyph_emb = rebuild_simple_model(token_ids = token_ids_ph, glyphs = glyph_ph, seq_lens = seq_lens, vocab_size = vocab_size, embed_dim = embed_dim, n_cnn_layers = n_cnn_layers, n_cnn_filters = n_cnn_filters, rnn_dim = rnn_dim, n_oov_buckets = n_oov_buckets)


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
                id_norm = np.linalg.norm(id_emb_val)
                glyph_norm = np.linalg.norm(glyph_emb_val)
                id_emb_accu += id_norm
                glyph_emb_accu += glyph_norm
                id_glyph_ratio_acc.append(id_norm / glyph_norm)


        id_emb_ave_norm = id_emb_accu / 4001.0
        glyph_emb_ave_norm = glyph_emb_accu / 4001.0

        print "id embedding average norm: ", id_emb_ave_norm
        print "glyph embedding average norm: ", glyph_emb_ave_norm

        np.savetxt('./ratio.txt', np.asarray(id_glyph_ratio_acc), delimiter=',')
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
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--rnn_dim', type=int, default=64)
    parser.add_argument('--task', type=str, default='seg')

    args = parser.parse_args()

    compute_embeddings(args.import_log_dir, args.dictionary, args.vocab_size, args.n_oov_buckets, args.embed_dim, args.rnn_dim, args.task)
