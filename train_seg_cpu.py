#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import os, glob

from render import render_text, ascii_print

class FastSaver(tf.train.Saver):
    # HACK disable saving metagraphs
    def save(self, sess, save_path, global_step=None, latest_filename=None, meta_graph_suffix='meta', write_meta_graph=True, write_state=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename, meta_graph_suffix, False, write_state)

def get_optimizer(opt_name, learning_rate, momentum):
    if opt_name == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif opt_name == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum, centered=True)
    else:
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=momentum)
    return optimizer

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
        num_outputs=2,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,
        scope='token_logit',
    )

    seq_logits = tf.reshape(token_logit, (bs, -1, 2))

    return seq_logits, final_state, rnn_input

def build_input_pipeline(text_corpus_path, target_corpus_path, vocabulary, batch_size, shuffle, allow_smaller_final_batch, num_epochs):
    # `text_corpus_path` could be a glob pattern
    input_filename_queue = tf.train.string_input_producer(glob.glob(text_corpus_path), shuffle=False, num_epochs=num_epochs)

    target_filename_queue = tf.train.string_input_producer(glob.glob(target_corpus_path), shuffle=False, num_epochs=num_epochs)

    _, input_line = tf.TextLineReader().read(input_filename_queue)
    _, target_line = tf.TextLineReader().read(target_filename_queue)

    seq_len = tf.shape(tf.string_split([input_line]).values)[0]

    if shuffle:
        batch = tf.train.shuffle_batch([input_line, seq_len, target_line], batch_size=batch_size, capacity=10 * batch_size, min_after_dequeue=5 * batch_size, allow_smaller_final_batch=allow_smaller_final_batch)
    else:
        batch = tf.train.batch([input_line, seq_len, target_line], batch_size=batch_size, capacity=10 * batch_size, allow_smaller_final_batch=allow_smaller_final_batch)

    tokens = tf.string_split(batch[0])
    mapping_strings = tf.constant(["0", "1"])
    ids = tf.sparse_tensor_to_dense(vocabulary.lookup(tokens), validate_indices=False)
    seq_lens = batch[1]
    target_tokens = tf.string_split(batch[2])

    targets = tf.sparse_tensor_to_dense(tf.contrib.lookup.string_to_index(
    target_tokens, mapping=mapping_strings, default_value=-1), validate_indices=False)

    return ids, seq_lens, targets

def metric_func(prediction, targets, mask):
    confusion_matrix = tf.confusion_matrix(
        labels = tf.reshape(targets, [-1]),
        predictions = tf.reshape(prediction, [-1]),
        num_classes=2,
        dtype=tf.float32,
        name="confusion_matrix",
        weights=tf.reshape(mask, [-1])
    )

    precision = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])

    recall = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])

    F1 = 2*(precision*recall)/(precision + recall)

    accuracy = (confusion_matrix[1][1] + confusion_matrix[0][0])/(tf.reduce_sum(confusion_matrix))

    return precision, recall, F1, accuracy

def train(train_split_path, train_tar_path, val_split_path, val_tar_path, dict_path, log_dir, batch_size, vocab_size, n_oov_buckets, initial_lr, lr_decay_steps, lr_decay_rate, lr_staircase, no_grad_clip, clip_norm, opt_method, momentum, val_interval):
    assert vocab_size >= 2, 'vocabulary has to include at least start_tag and end_tag'
    assert n_oov_buckets > 0, 'there must be at least 1 OOV bucket'

    # token to token-id lookup
    vocabulary = tf.contrib.lookup.string_to_index_table_from_file(dict_path, num_oov_buckets=n_oov_buckets, vocab_size=vocab_size)

    # input pipelines
    # train
    ids, seq_lens, targets = build_input_pipeline(train_split_path, train_tar_path, vocabulary, batch_size, shuffle=True, allow_smaller_final_batch=False, num_epochs=None)


    # validation
    val_ids, val_seq_lens, val_targets = build_input_pipeline(val_split_path, val_tar_path, vocabulary, 20 * batch_size, shuffle=False, allow_smaller_final_batch=False, num_epochs=None)
    

    # # model
    embed_dim, rnn_dim = 100, 64
    with tf.variable_scope('model'):
        seq_logits, final_state = build_model(ids, seq_lens, vocab_size + n_oov_buckets, embed_dim, rnn_dim)

    # validation
    with tf.variable_scope('model', reuse=True):
        val_seq_logits, val_final_state = build_model(val_ids, val_seq_lens, vocab_size + n_oov_buckets, embed_dim, rnn_dim)

    # # loss
    mask = tf.sequence_mask(seq_lens, dtype=tf.float32)
    loss = tf.contrib.seq2seq.sequence_loss(seq_logits, targets, mask, average_across_timesteps=True, average_across_batch=True)

    #TODO : review this part
    prediction = tf.argmax(seq_logits, axis = -1)

    precision, recall, F1, accuracy = metric_func(prediction, targets, mask)

    n_samples = tf.reduce_sum(mask)


    val_mask = tf.sequence_mask(val_seq_lens, dtype=tf.float32)
    val_loss = tf.contrib.seq2seq.sequence_loss(val_seq_logits, val_targets, val_mask, average_across_timesteps=True, average_across_batch=True)
    val_prediction = tf.argmax(val_seq_logits, axis = -1)

    val_precision, val_recall, val_F1, val_acc = metric_func(val_prediction, val_targets, val_mask)


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
    tf.summary.scalar('train/precision', precision)
    tf.summary.scalar('train/recall', recall)
    tf.summary.scalar('train/F1', F1)
    tf.summary.scalar('train/accuracy', accuracy)
    tf.summary.scalar('train/learning_rate', learning_rate)
    tf.summary.scalar('model/gradient_norm', grad_norm)
    tf.summary.scalar('model/clipped_gradient_norm', clipped_norm)
    tf.summary.scalar('model/variable_norm', var_norm)
    for g, v in zip(normed_grads, trainable_vars):
        tf.summary.histogram(v.name, g)

    train_summary_op = tf.summary.merge_all()

    val_summary = tf.summary.merge([
        tf.summary.scalar('val/loss', val_loss),
        tf.summary.scalar('val/perplexity', tf.exp(val_loss)),
        tf.summary.scalar('val/precision', val_precision),
        tf.summary.scalar('val/recall', val_recall),
        tf.summary.scalar('val/F1', val_F1),
        tf.summary.scalar('val/accuracy', val_acc)
    ])

    summary_dir = os.path.join(log_dir, 'logs')
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    best_model_checkpoint_dir = os.path.join(log_dir, 'best_model')

    writer = tf.summary.FileWriter(summary_dir, flush_secs=30)

    init_op = tf.global_variables_initializer()
    saver = FastSaver(var_list=tf.global_variables(), keep_checkpoint_every_n_hours=1, max_to_keep=2)
    # save metagraph once
    saver.export_meta_graph(os.path.join(checkpoint_dir, 'model.meta'))

    saver_best_model = FastSaver(var_list=tf.global_variables(), keep_checkpoint_every_n_hours = 1, max_to_keep = 10)
    saver_best_model.export_meta_graph(os.path.join(best_model_checkpoint_dir, 'model.meta'))

    sv = tf.train.Supervisor(
        is_chief=True,
        saver=saver,
        init_op=init_op,
        ready_op=tf.report_uninitialized_variables(tf.global_variables()),
        summary_writer=writer,
        summary_op=None,
        global_step=global_step,
        logdir=checkpoint_dir,
        save_model_secs=30,
        save_summaries_secs=30,
        )
    
    config = tf.ConfigProto(device_count = {'GPU':0}, intra_op_parallelism_threads=2) 
    #config = tf.ConfigProto(gpu_options={'allow_growth': True})

    minimum_loss_val = 10000.0

    with sv.managed_session('', config=config) as sess, sess.as_default():
        gs = global_step.eval()
        if gs == 0:
            print '* starting training at global step', gs
        else:
            print '* resuming training at global step', gs

        while not sv.should_stop():
            train_results = sess.run([train_summary_op, update_op])
            writer.add_summary(train_results[0], gs)
            gs = global_step.eval()
            if gs % val_interval == 0:
                loss_val, summary_val = sess.run([val_loss, val_summary])
                if loss_val < minimum_loss_val:
                    minimum_loss_val = loss_val
                    saver_best_model.save(sess, best_model_checkpoint_dir + '/model', write_meta_graph=False, global_step=gs)

                writer.add_summary(summary_val, gs)
                print 'step %i validation loss %g' % (gs, loss_val)

        sv.stop()

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

    parser.add_argument('-t', '--train-corpus', default='segmentation/pku_train_raw', help='path to the training split')
    parser.add_argument('--train-target', default='segmentation/pku_train_seg', help='path to the training segmentation answer' )
    parser.add_argument('-v', '--val-corpus', default='segmentation/pku_val_raw', help='path to the validation split')
    parser.add_argument('--val-target', default='segmentation/pku_val_seg')
    parser.add_argument('--dictionary', default='work/dict.txt', help='path to the dictionary file')
    parser.add_argument('--val-interval', type=int, default=10000, help='interval of evaluation on the validation split')

    args = parser.parse_args()

    train(args.train_corpus, args.train_target, args.val_corpus, args.val_target, args.dictionary, args.log_dir, args.batch_size, args.vocab_size, args.n_oov_buckets, args.initial_lr, args.lr_decay_steps, args.lr_decay_rate, args.lr_staircase, args.no_grad_clip, args.clip_norm, args.optimizer, args.momentum, args.val_interval)
