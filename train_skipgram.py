#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import tensorflow as tf
# from tensorflow.python import debug as tfdbg
import os, glob
from train_lm import get_optimizer, FastSaver
from render import render_text

def render_glyph(char, shape=(24, 24), font=None, interpolation=cv2.INTER_NEAREST):
    glyph = render_text(char, font)
    if np.prod(glyph.shape) == 0:
        return np.zeros(shape, dtype=np.int32)
    return cv2.resize(glyph, shape, interpolation=interpolation)

def generate_glyphs(ids_val, lines_val):
    batch_size, max_len = ids_val.shape
    glyphs = np.zeros((batch_size, max_len, 24, 24))
    for i, line in enumerate(lines_val):
        tokens = line.split()
        for j, token in enumerate(tokens):
            glyphs[i, j] = render_glyph(token.decode('utf8'), shape=(24, 24))
    return glyphs

def build_model(center_token_ids, context_token_ids, vocab_size, embed_dim, n_cnn_layers, n_cnn_filters, nce_noise_samples):
    # encoder
    net = tf.contrib.layers.embed_sequence([center_token_ids], vocab_size, embed_dim)

    net = tf.reshape(net, (-1, embed_dim))

    nce_weights = tf.get_variable('nce_weight', [vocab_size, embed_dim], 'float', tf.truncated_normal_initializer())
    nce_biases = tf.get_variable('nce_bias', [vocab_size], 'float', tf.zeros_initializer())

    nce_labels = tf.expand_dims(context_token_ids, -1)

    loss = tf.reduce_sum(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=nce_labels, inputs=net, num_sampled=nce_noise_samples, num_classes=vocab_size, remove_accidental_hits=True))

    return loss

def build_input_pipeline(corpus_path, vocabulary, batch_size, shuffle, allow_smaller_final_batch, num_epochs):
    # `corpus_path` could be a glob pattern
    filename_queue = tf.train.string_input_producer(glob.glob(corpus_path), shuffle=True, num_epochs=num_epochs)
    reader = tf.TextLineReader()
    _, line = reader.read(filename_queue)
    seq_len = tf.shape(tf.string_split([line]).values)[0]

    if shuffle:
        batch = tf.train.shuffle_batch([line, seq_len], batch_size=batch_size, capacity=10 * batch_size, min_after_dequeue=5 * batch_size, allow_smaller_final_batch=allow_smaller_final_batch)
    else:
        batch = tf.train.batch([line, seq_len], batch_size=batch_size, capacity=10 * batch_size, allow_smaller_final_batch=allow_smaller_final_batch)

    lines, seq_lens = batch
    tokens = tf.string_split(lines)
    ids = tf.sparse_tensor_to_dense(vocabulary.lookup(tokens), validate_indices=False)

    return ids, seq_lens, lines

def get_skip_pairs(ids, seq_lens, lines, max_window, batch_size):
    # yield batches of pairs (id(center_token), id(context_token))
    id_pairs = []
    # yield batches of pairs (center_token, context_token)
    token_pairs = []
    for i, line in enumerate(lines):
        tokens = line.split()
        for j, token in enumerate(tokens):
            center_token = token.decode('utf8')
            center_id = ids[i, j]
            # sample the window size
            # this effectively over-weights context tokens close to the center token
            half = np.random.randint(1, max_window + 1)
            for k in xrange(max(0, j - half), min(seq_lens[i], j + half + 1)):
                if k == j:
                    # skip the center token
                    continue
                context_token = tokens[k].decode('utf8')
                context_id = ids[i, k]

                id_pairs.append((center_id, context_id))
                token_pairs.append((center_token, context_token))
                if len(id_pairs) == batch_size:
                    yield id_pairs, token_pairs
                    id_pairs, token_pairs = [], []
    # return a possibly smaller final batch
    if len(id_pairs) > 0:
        yield id_pairs, token_pairs

def train(train_split_path, val_split_path, dict_path, log_dir, batch_size, vocab_size, n_oov_buckets, initial_lr, lr_decay_steps, lr_decay_rate, lr_staircase, no_grad_clip, clip_norm, opt_method, momentum, val_interval, save_interval, summary_interval):
    assert vocab_size >= 2, 'vocabulary has to include at least start_tag and end_tag'
    assert n_oov_buckets > 0, 'there must be at least 1 OOV bucket'

    # token to token-id lookup
    vocabulary = tf.contrib.lookup.string_to_index_table_from_file(dict_path, num_oov_buckets=n_oov_buckets, vocab_size=vocab_size)

    # input pipelines
    # train
    ids, seq_lens, lines = build_input_pipeline(train_split_path, vocabulary, batch_size, shuffle=True, allow_smaller_final_batch=False, num_epochs=None)

    # validation
    val_ids, val_seq_lens, val_lines = build_input_pipeline(val_split_path, vocabulary, batch_size, shuffle=False, allow_smaller_final_batch=False, num_epochs=None)

    # model
    embed_dim = 100
    n_cnn_layers, n_cnn_filters = 0, 32
    # glyph_ph = tf.placeholder('float', shape=[None, None, 24, 24], name='glyph')
    center_token_ids_ph = tf.placeholder('int32', shape=[None], name='center_token_id')
    context_token_ids_ph = tf.placeholder('int32', shape=[None], name='context_token_id')
    with tf.variable_scope('model'):
        loss = build_model(center_token_ids_ph, context_token_ids_ph, vocab_size + n_oov_buckets, embed_dim, n_cnn_layers, n_cnn_filters, 100)

    # validation
    # val_glyph_ph = tf.placeholder('float', [None, None, 24, 24], name='val_glyph')
    val_center_token_ids_ph = tf.placeholder('int32', shape=[None], name='val_center_token_id')
    val_context_token_ids_ph = tf.placeholder('int32', shape=[None], name='val_context_token_id')
    with tf.variable_scope('model', reuse=True):
        val_loss = build_model(val_center_token_ids_ph, val_context_token_ids_ph, vocab_size + n_oov_buckets, embed_dim, n_cnn_layers, n_cnn_filters, 100)

    # loss
    # loss = tf.losses.sparse_softmax_cross_entropy(context_token_ids, context_token_logits)
    # n_samples = tf.cast(tf.size(context_token_ids), 'float')

    # val_loss = tf.losses.sparse_softmax_cross_entropy(val_context_token_ids, val_context_token_logits)

    global_step = tf.contrib.framework.create_global_step()
    learning_rate = tf.train.exponential_decay(learning_rate=initial_lr, global_step=global_step, decay_steps=lr_decay_steps, decay_rate=lr_decay_rate, staircase=lr_staircase)
    optimizer = get_optimizer(opt_method, learning_rate, momentum)

    trainable_vars = tf.trainable_variables()
    grads = tf.gradients(loss, trainable_vars)
    grad_norm = tf.global_norm(grads)
    var_norm = tf.global_norm(trainable_vars)

    update_op = optimizer.apply_gradients(zip(grads, trainable_vars), global_step)

    # summaries
    tf.summary.scalar('train/loss', loss)
    # tf.summary.scalar('train/perplexity', tf.exp(loss))
    tf.summary.scalar('train/learning_rate', learning_rate)
    tf.summary.scalar('model/gradient_norm', grad_norm)
    tf.summary.scalar('model/variable_norm', var_norm)
    for g, v in zip(grads, trainable_vars):
        print g, v.name
        tf.summary.histogram(v.name, g)

    train_summary = tf.summary.merge_all()

    val_summary = tf.summary.merge([
        tf.summary.scalar('val/loss', val_loss),
        # tf.summary.scalar('val/perplexity', tf.exp(val_loss)),
    ])

    summary_dir = os.path.join(log_dir, 'logs')
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    writer = tf.summary.FileWriter(summary_dir, flush_secs=30)

    saver = FastSaver(var_list=tf.global_variables(), keep_checkpoint_every_n_hours=1, max_to_keep=2)
    # save metagraph once
    saver.export_meta_graph(os.path.join(checkpoint_dir, 'model.meta'))

    config = tf.ConfigProto(gpu_options={'allow_growth': True})
    with tf.Session(config=config) as sess:
        # sess = tfdbg.LocalCLIDebugWrapperSession(sess)

        checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        saver = tf.train.Saver(tf.global_variables())
        latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
        if not latest_checkpoint_path is None:
            print '* restoring model from checkpoint %s' % latest_checkpoint_path
            saver.restore(sess, latest_checkpoint_path)
        else:
            tf.global_variables_initializer().run()
        sess.run([tf.local_variables_initializer(), tf.tables_initializer()])

        # check uninitialized variables
        uninitialized_variables = tf.report_uninitialized_variables().eval()
        if len(uninitialized_variables) > 0:
            print '* some variables are not restored:'
            for v in uninitialized_variables:
                print v
            return None

        gs = global_step.eval()
        if gs == 0:
            print '* starting training at global step', gs
        else:
            print '* resuming training at global step', gs

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop():
                ids_val, seq_lens_val, lines_val = sess.run([ids, seq_lens, lines])

                # generate glyphs for characters
                # glyphs = generate_glyphs(ids_val, lines_val)

                for id_pairs, _ in get_skip_pairs(ids_val, seq_lens_val, lines_val, 10, 100):
                    id_pairs = np.asarray(id_pairs)

                    feed = {
                        center_token_ids_ph: id_pairs[:, 0],
                        context_token_ids_ph: id_pairs[:, 1],
                    }

                    if gs % summary_interval == 0:
                        _, gs, train_summary_val = sess.run([update_op, global_step, train_summary], feed_dict=feed)
                        writer.add_summary(train_summary_val, gs)
                    else:
                        _, gs = sess.run([update_op, global_step], feed_dict=feed)

                    if gs % save_interval == 0:
                        saver.save(sess, checkpoint_dir + '/model', write_meta_graph=False, global_step=gs)

                    if gs % val_interval == 0:
                        val_ids_val, val_seq_lens_val, val_lines_val = sess.run([val_ids, val_seq_lens, val_lines])

                        # generate glyphs for characters
                        # val_glyphs = generate_glyphs(val_ids_val, val_lines_val)
                        val_id_pairs, _ = get_skip_pairs(val_ids_val, val_seq_lens_val, val_lines_val, 10, 100).next()
                        val_id_pairs = np.asarray(val_id_pairs)

                        loss_val, val_summary_val = sess.run([val_loss, val_summary], feed_dict={
                            val_center_token_ids_ph: val_id_pairs[:, 0],
                            val_context_token_ids_ph: val_id_pairs[:, 1],
                        })
                        writer.add_summary(val_summary_val, gs)
                        print 'step %i validation loss %g' % (gs, loss_val)
            coord.join(threads)
        except tf.errors.OutOfRangeError:
            print 'epoch limit reached'
        except Exception as e:
            coord.request_stop(e)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log-dir', required=True)
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-m', '--vocab-size', type=int, default=1000)
    parser.add_argument('-o', '--n-oov-buckets', type=int, default=1)

    parser.add_argument('--optimizer', choices=['adam', 'rmsprop', 'momentum'], default='adam')
    parser.add_argument('--momentum', type=float, default=0.)
    parser.add_argument('--initial-lr', type=float, default=1e-3)
    parser.add_argument('--lr-decay-rate', type=float, default=1.)
    parser.add_argument('--lr-decay-steps', type=int, default=10**5)
    parser.add_argument('--lr-staircase', action='store_true')
    parser.add_argument('--no-grad-clip', action='store_true', help='disable gradient clipping')
    parser.add_argument('--clip-norm', type=float, default=40.)

    parser.add_argument('-t', '--train-corpus', default='work/train.txt', help='path to the training split')
    parser.add_argument('-v', '--val-corpus', default='work/val.txt', help='path to the validation split')
    parser.add_argument('--dictionary', default='work/dict.txt', help='path to the dictionary file')
    parser.add_argument('--save-interval', type=int, default=32, help='interval of checkpoints')
    parser.add_argument('--summary-interval', type=int, default=32, help='interval of summary')
    parser.add_argument('--val-interval', type=int, default=64, help='interval of evaluation on the validation split')

    args = parser.parse_args()

    train(args.train_corpus, args.val_corpus, args.dictionary, args.log_dir, args.batch_size, args.vocab_size, args.n_oov_buckets, args.initial_lr, args.lr_decay_steps, args.lr_decay_rate, args.lr_staircase, args.no_grad_clip, args.clip_norm, args.optimizer, args.momentum, args.val_interval, args.save_interval, args.summary_interval)
