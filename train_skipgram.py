#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
# from tensorflow.python import debug as tfdbg
import os, glob
from train_lm import get_optimizer
from train_cnn_lm import build_input_pipeline, render_glyph

def build_model(center_token_ids, context_token_ids, vocab_size, embed_dim, n_cnn_layers, n_cnn_filters, nce_noise_samples):
    # encoder
    embeddings = tf.get_variable('embeddings', [vocab_size, embed_dim], 'float', tf.random_uniform_initializer(-1., 1.))

    net = tf.nn.embedding_lookup(embeddings, center_token_ids)

    nce_weights = tf.get_variable('nce_weight', [vocab_size, embed_dim], 'float', tf.truncated_normal_initializer())
    nce_biases = tf.get_variable('nce_bias', [vocab_size], 'float', tf.zeros_initializer())

    nce_labels = tf.expand_dims(context_token_ids, -1)

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=nce_labels, inputs=net, num_sampled=nce_noise_samples, num_classes=vocab_size, remove_accidental_hits=False))

    return embeddings, loss

def build_glyph_aware_model(center_token_ids, center_token_glyphs, context_token_ids, vocab_size, embed_dim, n_cnn_layers, n_cnn_filters, nce_noise_samples):
    # encoder
    embeddings = tf.get_variable('embeddings', [vocab_size, embed_dim], 'float', tf.random_uniform_initializer(-1., 1.))

    glyph_unaware = tf.nn.embedding_lookup(embeddings, center_token_ids)

    # glyph CNN
    net = center_token_glyphs / 255.
    net = tf.expand_dims(net, -1)
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

    net = tf.contrib.layers.convolution2d(
        inputs=net,
        num_outputs=16,
        kernel_size=(5, 5),
        stride=(2, 2),
        activation_fn=tf.nn.relu,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        scope='conv2',
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

    net += glyph_unaware

    # noise contrastive estimation (output are sigmoids)
    nce_weights = tf.get_variable('nce_weight', [vocab_size, embed_dim], 'float', tf.truncated_normal_initializer())
    nce_biases = tf.get_variable('nce_bias', [vocab_size], 'float', tf.zeros_initializer())

    nce_labels = tf.expand_dims(context_token_ids, -1)

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=nce_labels, inputs=net, num_sampled=nce_noise_samples, num_classes=vocab_size, remove_accidental_hits=False))

    return embeddings, loss

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

def generate_glyphs(batch_size, token_pairs, glyph_width):
    glyphs = np.zeros((batch_size, glyph_width, glyph_width))
    for i, (center_token, _) in enumerate(token_pairs):
        glyphs[i] = render_glyph(center_token)
    return glyphs

def train(train_split_path, val_split_path, dict_path, log_dir, batch_size, vocab_size, n_oov_buckets, n_noisy_samples, initial_lr, lr_decay_steps, lr_decay_rate, lr_staircase, no_grad_clip, clip_norm, opt_method, momentum, val_interval, save_interval, summary_interval):
    assert vocab_size >= 2, 'vocabulary has to include at least start_tag and end_tag'
    assert n_oov_buckets > 0, 'there must be at least 1 OOV bucket'

    # token to token-id lookup
    vocabulary = tf.contrib.lookup.string_to_index_table_from_file(dict_path, num_oov_buckets=n_oov_buckets, vocab_size=vocab_size)

    # input pipelines
    # train
    queue_batch = 4
    ids, seq_lens, lines = build_input_pipeline(train_split_path, vocabulary, queue_batch, shuffle=True, allow_smaller_final_batch=False, num_epochs=None)

    # validation
    val_ids, val_seq_lens, val_lines = build_input_pipeline(val_split_path, vocabulary, queue_batch, shuffle=False, allow_smaller_final_batch=False, num_epochs=None)

    # model
    embed_dim = 100
    glyph_width = 24
    n_cnn_layers, n_cnn_filters = 0, 32
    center_token_glyphs_ph = tf.placeholder('float', shape=[None, glyph_width, glyph_width], name='glyph')
    center_token_ids_ph = tf.placeholder('int32', shape=[None], name='center_token_id')
    context_token_ids_ph = tf.placeholder('int32', shape=[None], name='context_token_id')
    with tf.variable_scope('model'):
        embeddings, loss = build_glyph_aware_model(center_token_ids_ph, center_token_glyphs_ph, context_token_ids_ph, vocab_size + n_oov_buckets, embed_dim, n_cnn_layers, n_cnn_filters, n_noisy_samples)

    # validation
    val_center_token_glyphs_ph = tf.placeholder('float', [None, glyph_width, glyph_width], name='val_glyph')
    val_center_token_ids_ph = tf.placeholder('int32', shape=[None], name='val_center_token_id')
    val_context_token_ids_ph = tf.placeholder('int32', shape=[None], name='val_context_token_id')
    with tf.variable_scope('model', reuse=True):
        _, val_loss = build_glyph_aware_model(val_center_token_ids_ph, val_center_token_glyphs_ph, val_context_token_ids_ph, vocab_size + n_oov_buckets, embed_dim, n_cnn_layers, n_cnn_filters, n_noisy_samples)

    # loss
    n_samples = tf.cast(tf.size(context_token_ids_ph), 'float')

    global_step = tf.contrib.framework.create_global_step()
    learning_rate = tf.train.exponential_decay(learning_rate=initial_lr, global_step=global_step, decay_steps=lr_decay_steps, decay_rate=lr_decay_rate, staircase=lr_staircase)
    optimizer = get_optimizer(opt_method, learning_rate, momentum)

    trainable_vars = tf.trainable_variables()
    grads = tf.gradients(loss * n_samples, trainable_vars)
    grad_norm = tf.global_norm(grads)
    var_norm = tf.global_norm(trainable_vars)

    update_op = optimizer.apply_gradients(zip(grads, trainable_vars), global_step)

    # summaries
    tf.summary.scalar('train/loss', loss)
    tf.summary.scalar('train/perplexity', tf.exp(loss))
    tf.summary.scalar('train/learning_rate', learning_rate)
    tf.summary.scalar('model/gradient_norm', grad_norm)
    tf.summary.scalar('model/variable_norm', var_norm)
    for g, v in zip(grads, trainable_vars):
        tf.summary.histogram(v.name, g)

    train_summary = tf.summary.merge_all()

    val_summary = tf.summary.merge([
        tf.summary.scalar('val/loss', val_loss),
        tf.summary.scalar('val/perplexity', tf.exp(val_loss)),
    ])

    summary_dir = os.path.join(log_dir, 'logs')
    # XXX putting summary and checkpoints into one folder due to TF issue
    # checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    checkpoint_dir = summary_dir

    saver = tf.train.Saver(var_list=tf.global_variables(), keep_checkpoint_every_n_hours=1, max_to_keep=2)
    # save metagraph once
    saver.export_meta_graph(os.path.join(checkpoint_dir, 'model.meta'))

    # visualize embeddings
    # XXX the `model_checkpoint_dir` seems to be useless
    projector_config = projector.ProjectorConfig()
    embed = projector_config.embeddings.add()
    embed.tensor_name = embeddings.name
    # generate the metadata file if missing
    metadata_path = os.path.join(summary_dir, 'projector_metadata.txt')
    if not os.path.exists(metadata_path):
        with open(metadata_path, 'wb') as fp:
            with open(dict_path) as dict_fp:
                for _ in xrange(vocab_size):
                    fp.write(dict_fp.next())
                for i in xrange(n_oov_buckets):
                    # buckets for unknown tokens
                    fp.write('<UNK%i>\n' % i)
        print '* created metadata file for tensorboard projector at %s' % metadata_path
    else:
        print '* found metadata file for tensorboard projector at %s' % metadata_path
    embed.metadata_path = metadata_path

    config = tf.ConfigProto(gpu_options={'allow_growth': True})
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter(summary_dir, flush_secs=30, graph=sess.graph)
        projector.visualize_embeddings(writer, projector_config)
        # sess = tfdbg.LocalCLIDebugWrapperSession(sess)

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

                for id_pairs, token_pairs in get_skip_pairs(ids_val, seq_lens_val, lines_val, 10, batch_size):
                    id_pairs = np.asarray(id_pairs)

                    # generate glyphs for characters
                    glyphs = generate_glyphs(len(id_pairs), token_pairs, glyph_width)

                    feed = {
                        center_token_ids_ph: id_pairs[:, 0],
                        context_token_ids_ph: id_pairs[:, 1],
                        center_token_glyphs_ph: glyphs,
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

                        val_id_pairs, val_token_pairs = get_skip_pairs(val_ids_val, val_seq_lens_val, val_lines_val, 10, batch_size).next()
                        val_id_pairs = np.asarray(val_id_pairs)
                        # generate glyphs for characters
                        glyphs = generate_glyphs(len(val_id_pairs), val_token_pairs, glyph_width)

                        loss_val, val_summary_val = sess.run([val_loss, val_summary], feed_dict={
                            val_center_token_ids_ph: val_id_pairs[:, 0],
                            val_context_token_ids_ph: val_id_pairs[:, 1],
                            val_center_token_glyphs_ph: glyphs,
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
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-s', '--n-noisy-samples', type=int, default=100)
    parser.add_argument('-m', '--vocab-size', type=int, default=1000)
    parser.add_argument('-o', '--n-oov-buckets', type=int, default=1)

    parser.add_argument('--optimizer', choices=['adam', 'rmsprop', 'momentum'], default='adam')
    parser.add_argument('--momentum', type=float, default=0.)
    parser.add_argument('--initial-lr', type=float, default=1e-3)
    parser.add_argument('--lr-decay-rate', type=float, default=0.99)
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

    train(args.train_corpus, args.val_corpus, args.dictionary, args.log_dir, args.batch_size, args.vocab_size, args.n_oov_buckets, args.n_noisy_samples, args.initial_lr, args.lr_decay_steps, args.lr_decay_rate, args.lr_staircase, args.no_grad_clip, args.clip_norm, args.optimizer, args.momentum, args.val_interval, args.save_interval, args.summary_interval)
