#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import tensorflow as tf
# from tensorflow.python import debug as tfdbg
import os, glob
from train_lm import get_optimizer, FastSaver
from train_cnn_lm import render_glyph
from models.glyph_embed import simple_cnn_1, simple_cnn_2, multi_path_cnn_1, linear

def generate_glyphs(ids_val, lines_val):
    batch_size, max_len = ids_val.shape
    glyphs = np.zeros((batch_size, max_len, 24, 24))
    for i, line in enumerate(lines_val):
        tokens = line.split()
        for j, token in enumerate(tokens):
            glyphs[i, j] = render_glyph(token.decode('utf8'))
    return glyphs

def build_model(token_ids, glyphs, seq_lens, vocab_size, n_oov_buckets, embed_dim, rnn_dim, train_test_mode = 'train', id_glyph_switch = 'mix'):
    # encoder
    bs = tf.size(seq_lens)
    glyph_unaware = tf.contrib.layers.embed_sequence(token_ids, vocab_size + n_oov_buckets, embed_dim)
    # glyph-aware
    glyphs = tf.reshape(glyphs, (-1, 24, 24, 1))
    # linear glyph embedder
    net = linear.build_model(glyphs, embed_dim)

    glyph_aware = tf.reshape(net, (bs, -1, embed_dim))
    output_vocab_size = 2000
    in_vocab = tf.expand_dims(tf.cast(tf.less(token_ids, vocab_size), tf.float32), -1)
    glyph_unk = tf.Variable(np.random.rand(1, embed_dim), name = 'glyph_unk', dtype = tf.float32)

    glyph_unk_pad = (1 - in_vocab) * (tf.zeros_like(glyph_unaware) + glyph_unk)
    # msr-m1, msr-m0, msr-l0
    # rnn_input = glyph_unaware + in_vocab * glyph_aware
    # msr-i0
    # rnn_input = glyph_unaware + 0. * glyph_aware
    # msr-l1, msr-c2
    rnn_input = None
    if id_glyph_switch == 'mix':
        if train_test_mode == 'train':

            rnn_input = glyph_unaware + in_vocab * glyph_aware + (1 - in_vocab) * glyph_unk_pad
        else:
            rnn_input = glyph_unaware + glyph_aware

    elif id_glyph_switch == 'id':
        rnn_input = glyph_unaware

    elif id_glyph_switch == 'glyph':
        if train_test_mode == 'train':
            rnn_input = in_vocab * glyph_aware + (1 - in_vocab) * glyph_unk_pad
        else:
            rnn_input = glyph_aware

    # rnn
    cell = tf.contrib.rnn.GRUBlockCell(rnn_dim)
    rnn_output, final_state = tf.nn.dynamic_rnn(cell, rnn_input, seq_lens, cell.zero_state(bs, 'float'))

    # decoder
    decoder_input = tf.reshape(rnn_output, (-1, rnn_dim))
    token_logit = tf.contrib.layers.fully_connected(
        inputs=decoder_input,
        num_outputs=output_vocab_size,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,
        scope='token_logit',
    )

    seq_logits = tf.reshape(token_logit, (bs, -1, output_vocab_size))

    return seq_logits, final_state

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

def train(train_split_path, val_split_path, dict_path, log_dir, batch_size, vocab_size, n_oov_buckets, initial_lr, lr_decay_steps, lr_decay_rate, lr_staircase, no_grad_clip, clip_norm, opt_method, momentum, val_interval, save_interval, summary_interval, id_glyph_switch):
    assert vocab_size >= 2, 'vocabulary has to include at least start_tag and end_tag'
    assert n_oov_buckets > 0, 'there must be at least 1 OOV bucket'

    # token to token-id lookup
    vocabulary = tf.contrib.lookup.string_to_index_table_from_file(dict_path, num_oov_buckets=n_oov_buckets, vocab_size=vocab_size)

    # input pipelines
    # train
    ids, seq_lens, lines = build_input_pipeline(train_split_path, vocabulary, batch_size, shuffle=True, allow_smaller_final_batch=False, num_epochs=None)
    seq_lens -= 1

    # validation
    val_ids, val_seq_lens, val_lines = build_input_pipeline(val_split_path, vocabulary, batch_size, shuffle=False, allow_smaller_final_batch=False, num_epochs=None)
    val_seq_lens -= 1

    # model
    embed_dim, rnn_dim = 300, 128
    glyph_ph = tf.placeholder('float', shape=[None, None, 24, 24], name='glyph')
    with tf.variable_scope('model'):
        seq_logits, final_state = build_model(ids[:, :-1], glyph_ph[:, :-1], seq_lens, vocab_size, n_oov_buckets, embed_dim, rnn_dim, 'train', id_glyph_switch)

    # validation
    val_glyph_ph = tf.placeholder('float', [None, None, 24, 24], name='val_glyph')
    with tf.variable_scope('model', reuse=True):
        val_seq_logits, val_final_state = build_model(val_ids[:, :-1], val_glyph_ph[:, :-1], val_seq_lens, vocab_size, n_oov_buckets, embed_dim, rnn_dim, 'test', id_glyph_switch)

    # loss
    mask = tf.sequence_mask(seq_lens, dtype=tf.float32)
    loss = tf.contrib.seq2seq.sequence_loss(seq_logits, ids[:, 1:], mask, average_across_timesteps=True, average_across_batch=True)
    n_samples = tf.reduce_sum(mask)

    val_mask = tf.sequence_mask(val_seq_lens, dtype=tf.float32)
    val_loss = tf.contrib.seq2seq.sequence_loss(val_seq_logits, val_ids[:, 1:], val_mask, average_across_timesteps=True, average_across_batch=True)

    global_step = tf.contrib.framework.create_global_step()
    learning_rate = tf.train.exponential_decay(learning_rate=initial_lr, global_step=global_step, decay_steps=lr_decay_steps, decay_rate=lr_decay_rate, staircase=lr_staircase)
    optimizer = get_optimizer(opt_method, learning_rate, momentum)

    trainable_vars = tf.trainable_variables()
    # re-scale the loss to be the sum of losses of all time steps
    grads = tf.gradients(loss * n_samples, trainable_vars)
    grad_norm = tf.global_norm(grads)
    var_norm = tf.global_norm(trainable_vars)

    if no_grad_clip:
        normed_grads = grads
        clipped_norm = grad_norm
    else:
        # gradient clipping
        normed_grads, _ = tf.clip_by_global_norm(grads, clip_norm, grad_norm)
        clipped_norm = tf.minimum(clip_norm, grad_norm)

    update_op = optimizer.apply_gradients(zip(normed_grads, trainable_vars), global_step)

    # summaries
    tf.summary.scalar('train/loss', loss)
    tf.summary.scalar('train/perplexity', tf.exp(loss))
    tf.summary.scalar('train/learning_rate', learning_rate)
    tf.summary.scalar('model/gradient_norm', grad_norm)
    tf.summary.scalar('model/clipped_gradient_norm', clipped_norm)
    tf.summary.scalar('model/variable_norm', var_norm)
    # for g, v in zip(normed_grads, trainable_vars):
    #     tf.summary.histogram(v.name, g)

    train_summary = tf.summary.merge_all()

    val_summary = tf.summary.merge([
        tf.summary.scalar('val/loss', val_loss),
        tf.summary.scalar('val/perplexity', tf.exp(val_loss)),
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
                glyphs = generate_glyphs(ids_val, lines_val)

                feed = {
                    ids: ids_val,
                    seq_lens: seq_lens_val,
                    glyph_ph: glyphs,
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
                    val_glyphs = generate_glyphs(val_ids_val, val_lines_val)

                    loss_val, val_summary_val = sess.run([val_loss, val_summary], feed_dict={
                        val_ids: val_ids_val,
                        val_seq_lens: val_seq_lens_val,
                        val_glyph_ph: val_glyphs,
                    })
                    writer.add_summary(val_summary_val, gs)
                    print 'step %i validation loss %g' % (gs, loss_val)
            coord.join(threads)
        except tf.errors.OutOfRangeError:
            print 'epoch limit reached'

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
    parser.add_argument('--id_glyph_switch', type=str, default='id', help='decide which type of input to use')
    
    args = parser.parse_args()

    train(args.train_corpus, args.val_corpus, args.dictionary, args.log_dir, args.batch_size, args.vocab_size, args.n_oov_buckets, args.initial_lr, args.lr_decay_steps, args.lr_decay_rate, args.lr_staircase, args.no_grad_clip, args.clip_norm, args.optimizer, args.momentum, args.val_interval, args.save_interval, args.summary_interval, args.id_glyph_switch)
