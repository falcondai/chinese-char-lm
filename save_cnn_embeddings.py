#!/usr/bin/env python

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
# from tensorflow.python import debug as tfdbg
import os, glob
from train_cnn_lm import render_glyph
from train_skipgram import build_glyph_aware_model

def compute_embeddings(checkpoint_dir, embeddings_checkpoint_dir, dict_path, vocab_size, n_oov_buckets, embed_dim):
    glyph_width = 24
    center_token_glyphs_ph = tf.placeholder('float', shape=[None, glyph_width, glyph_width], name='glyph')
    center_token_ids_ph = tf.placeholder('int32', shape=[None], name='center_token_id')
    context_token_ids_ph = tf.placeholder('int32', shape=[None], name='context_token_id')
    with tf.variable_scope('model'):
        embedding, glyph_unaware, _ = build_glyph_aware_model(center_token_ids_ph, center_token_glyphs_ph, context_token_ids_ph, vocab_size + n_oov_buckets, embed_dim, 0, 0, 1)

    saver = tf.train.Saver(var_list=tf.global_variables(), keep_checkpoint_every_n_hours=1, max_to_keep=2)
    embeddings = tf.get_variable('embeddings', [vocab_size + n_oov_buckets, embed_dim], 'float', tf.zeros_initializer(), trainable=False)
    embeddings_ph = tf.placeholder('float', [vocab_size + n_oov_buckets, embed_dim])
    embeddings_assign_op = embeddings.assign(embeddings_ph)
    embed_saver = tf.train.Saver([embeddings])

    if not os.path.exists(embeddings_checkpoint_dir):
        os.makedirs(embeddings_checkpoint_dir)

    # visualize embeddings
    projector_config = projector.ProjectorConfig()
    embed = projector_config.embeddings.add()
    embed.tensor_name = embeddings.name
    metadata_path = os.path.join(embeddings_checkpoint_dir, 'projector_metadata.txt')
    embed.metadata_path = metadata_path

    config = tf.ConfigProto(gpu_options={'allow_growth': True})
    with tf.Session(config=config) as sess:
        latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
        print '* restoring model from checkpoint %s' % latest_checkpoint_path
        saver.restore(sess, latest_checkpoint_path)

        embeddings_val = np.zeros((vocab_size + n_oov_buckets, embed_dim))
        with open(dict_path) as fp:
            with open(metadata_path, 'wb') as mp:
                # headers
                mp.write('%s\t%s\n' % ('token', 'id'))
                # vocabulary tokens
                for i, token in enumerate(fp):
                    if i == vocab_size:
                        break
                    token = token.strip().decode('utf8')
                    embedding_val = embedding.eval({
                        center_token_ids_ph: [i],
                        center_token_glyphs_ph: [render_glyph(token)],
                    })
                    embeddings_val[i] = embedding_val[0]
                    mp.write('%s\t%i\n' % (token.encode('utf8'), i))
                # OOV buckets
                for i in xrange(n_oov_buckets):
                    embedding_val = glyph_unaware.eval({
                        center_token_ids_ph: [i],
                    })
                    embeddings_val[vocab_size + i] = embedding_val[0]
                    mp.write('<UNK%i>\t%i\n' % (i, vocab_size + i))

            print '* created metadata file for tensorboard projector at %s' % metadata_path
        writer = tf.summary.FileWriter(embeddings_checkpoint_dir, flush_secs=30)
        projector.visualize_embeddings(writer, projector_config)

        sess.run(embeddings_assign_op, {
            embeddings_ph: embeddings_val,
        })
        embed_saver.save(sess, os.path.join(embeddings_checkpoint_dir, 'embeddings'))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--export-log-dir', required=True, help='path to save the embeddings')
    parser.add_argument('-l', '--import-log-dir', required=True, help='path to skip-gram model checkpoint directory')
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-m', '--vocab-size', type=int, default=4000)
    parser.add_argument('-o', '--n-oov-buckets', type=int, default=1)
    parser.add_argument('--embed-dim', type=int, default=100)
    parser.add_argument('--dictionary', default='work/dict.txt', help='path to the dictionary file')

    args = parser.parse_args()

    compute_embeddings(args.import_log_dir, args.export_log_dir, args.dictionary, args.vocab_size, args.n_oov_buckets, args.embed_dim)
