#!/usr/bin/env python

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
# from tensorflow.python import debug as tfdbg
import os, glob
from train_id_cnn_lm import generate_glyphs, render_glyph

def rebuild_model(token_ids, glyphs, seq_lens, vocab_size, embed_dim, n_cnn_layers, n_cnn_filters, wemb):
    # encoder
    bs = tf.size(seq_lens)
    wemb = wemb
    glyph_unaware = tf.nn.embedding_lookup(wemb, token_ids)

    # glyph-aware
    net = glyphs / 255.
    net = tf.reshape(net, (-1, 24, 24, 1))
    net = tf.contrib.layers.convolution2d(
        inputs=net,
        num_outputs=32,
        kernel_size=(7, 7),
        stride=(2, 2),
        activation_fn=tf.nn.relu,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        scope='conv1',
    )

    # net = tf.contrib.layers.convolution2d(
    #     inputs=net,
    #     num_outputs=16,
    #     kernel_size=(5, 5),
    #     stride=(2, 2),
    #     activation_fn=tf.nn.relu,
    #     biases_initializer=tf.zeros_initializer(),
    #     weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
    #     scope='conv2',
    # )

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

    return glyph_unglyph_aware

def compute_embeddings(checkpoint_dir, dict_path, vocab_size, n_oov_buckets, embed_dim):
    glyph_width = 24

    token_id_ph = tf.placeholder('int32', shape=[None], name='center_token_id')

    # model
    glyph_ph = tf.placeholder('float', shape=[None, None, 24, 24], name='glyph')
    # token to token-id lookup

    seq_lens = tf.stack([tf.shape(token_ids_ph)[0]])

    id_token_vocabulary = tf.contrib.lookup.index_to_string_table_from_file(dict_path, vocab_size=vocab_size)

    embed_dim, rnn_dim = 100, 64
    n_cnn_layers, n_cnn_filters = 0, 32
    saver = tf.train.Saver()
    seq_lens = [1]
    with tf.variable_scope('model'):
        id_emb, glyph_emb = rebuild_model(token_ids = token_id_ph, glyphs = glyph_ph, seq_lens = seq_lens, vocab_size = vocab_size, embed_dim = embed_dim, n_cnn_layers = n_cnn_layers, n_cnn_filters = n_cnn_filters)

    # if not os.path.exists(embeddings_checkpoint_dir):
    #     os.makedirs(embeddings_checkpoint_dir)

    # # visualize embeddings
    # projector_config = projector.ProjectorConfig()
    # embed = projector_config.embeddings.add()
    # embed.tensor_name = embeddings.name
    # metadata_path = os.path.join(embeddings_checkpoint_dir, 'projector_metadata.txt')
    # embed.metadata_path = metadata_path 
    # Load the VGG-16 model in the default graph
    latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    print '* restoring model from checkpoint %s' % latest_checkpoint_path
    saver.restore(sess, latest_checkpoint_path)

    saver = tf.train.import_meta_graph(checkpoint_dir)
    # Access the graph
    graph = tf.get_default_graph()

    # Retrieve variables
    conv_bias = graph.get_tensor_by_name('model/conv1/biases:0')
    conv_weight = graph.get_tensor_by_name('model/conv1/weights:0')
    wemb_fc_weight = graph.get_tensor_by_name('model/embedding_fc/weights:0')
    wemb_fc_bias = graph.get_tensor_by_name('model/embedding_fc/biases:0')
    wemb_matrix = graph.get_tensor_by_name('model/EmbedSequence/embeddings:0')
    
    config = tf.ConfigProto(gpu_options={'allow_growth': True})
    with tf.Session(config=config) as sess:

        id_token_dict = id_token_vocabulary.eval()

        embeddings_val = np.zeros((vocab_size + n_oov_buckets, embed_dim))
        for i in range(vocab_size):
            token = sess.run([])
            feed_dictionary = {token_ids_ph:i, glyph_ph:render_glyph(id_token_dict[i].decode('utf8'), shape=(24, 24))}

            sess.run([id_emb, glyph_emb], feed_dict=feed_dictionary)
            print id_emb, glyph_emb
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
