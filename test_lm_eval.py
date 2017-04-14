#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
# XXX importing debug module seems to cause segfault
# from tensorflow.python import debug as tfdbg
import os, glob

from train_lm_eval import build_input_pipeline, build_model

def test(test_split_path, dict_path, embedding_path, log_dir, batch_size, vocab_size, n_oov_buckets, print_interval):
    # token to token-id lookup
    vocabulary = tf.contrib.lookup.string_to_index_table_from_file(dict_path, num_oov_buckets=n_oov_buckets, vocab_size=vocab_size)

    # input pipelines
    # test
    ids, seq_lens = build_input_pipeline(test_split_path, vocabulary, batch_size, shuffle=False, allow_smaller_final_batch=True, num_epochs=1)
    seq_lens -= 1

    # model
    embed_dim, rnn_dim = 100, 64
    # trained_embeddings = tf.contrib.framework.load_variable(embedding_path, 'embeddings')
    trained_embeddings = tf.contrib.framework.load_variable(embedding_path, 'model/EmbedSequence/embeddings')
    embeddings = tf.get_variable('embeddings', (vocab_size + n_oov_buckets, embed_dim), 'float', trainable=False, initializer=tf.constant_initializer(trained_embeddings))
    with tf.variable_scope('model'):
        seq_logits, final_state = build_model(ids[:, :-1], embeddings, seq_lens, vocab_size + n_oov_buckets, rnn_dim)

    # loss
    mask = tf.sequence_mask(seq_lens, dtype=tf.float32)
    loss = tf.contrib.seq2seq.sequence_loss(seq_logits, ids[:, 1:], mask, average_across_timesteps=True, average_across_batch=True)
    n_samples = tf.reduce_sum(mask)

    config = tf.ConfigProto(gpu_options={'allow_growth': True})
    with tf.Session(config=config) as sess:
        # to debug
        # sess = tfdbg.LocalCLIDebugWrapperSession(sess)

        checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        saver = tf.train.Saver(tf.global_variables())
        latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
        print '* restoring model from checkpoint %s' % latest_checkpoint_path
        saver.restore(sess, latest_checkpoint_path)

        # initialize the token-id lookup table and local variables
        # before starting the queue runners
        sess.run([tf.local_variables_initializer(), tf.tables_initializer()])

        # check uninitialized variables
        uninitialized_variables = tf.report_uninitialized_variables().eval()
        if len(uninitialized_variables) > 0:
            print '* some variables are not restored:'
            for v in uninitialized_variables:
                print v
            return None

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        total_loss = 0.
        total_n = 0
        i = 0
        try:
            while not coord.should_stop():
                n, loss_val = sess.run([n_samples, loss])
                total_loss += n * loss_val
                total_n += n
                if i % print_interval == 0:
                    print 'minibatch %i loss %g' % (i, loss_val)
                i += 1
            coord.join(threads)
        except tf.errors.OutOfRangeError:
            print 'epoch limit reached'
        except Exception as e:
            coord.request_stop(e)

    per_symbol_loss = total_loss / total_n
    print '* %i samples test loss %g perplexity %g' % (total_n, per_symbol_loss, np.exp(per_symbol_loss))
    return per_symbol_loss

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log-dir', required=True)
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-m', '--vocab-size', type=int, default=1000)
    parser.add_argument('-o', '--n-oov-buckets', type=int, default=1)
    parser.add_argument('-e', '--embedding', default='work/sg4k-1/logs', help='path to the checkpoint dir (or path) containing trained embeddings')
    parser.add_argument('-t', '--test-corpus', default='work/test.txt', help='path to the test split')
    parser.add_argument('--dictionary', default='work/dict.txt', help='path to the dictionary file')
    parser.add_argument('--print-interval', type=int, default=16, help='interval of printing minibatch evaluation results')

    args = parser.parse_args()

    test(args.test_corpus, args.dictionary, args.embedding, args.log_dir, args.batch_size, args.vocab_size, args.n_oov_buckets, args.print_interval)
