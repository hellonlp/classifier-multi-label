# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 19:25:55 2020

@author: cm
"""



import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import numpy as np
import tensorflow as tf

from classifier_multi_label_seq2seq.networks import NetworkAlbertSeq2Seq
from classifier_multi_label_seq2seq.classifier_utils import get_features
from classifier_multi_label_seq2seq.hyperparameters import Hyperparamters as hp
from classifier_multi_label_seq2seq.utils import select, shuffle_one, time_now_string
from classifier_multi_label_seq2seq.load import normalization_label


pwd = os.path.dirname(os.path.abspath(__file__))
MODEL = NetworkAlbertSeq2Seq(is_training=True)


# Get data features
input_ids, input_masks, segment_ids, label_ids = get_features()
num_train_samples = len(input_ids)
arr = np.arange(num_train_samples)
num_batchs = int((num_train_samples - 1) / hp.batch_size) + 1
print('Number of batch:', num_batchs)

# Set up the graph
saver = tf.train.Saver(max_to_keep=hp.max_to_keep)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Load model saved before
MODEL_SAVE_PATH = os.path.join(pwd, hp.file_model_save)
ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Restored model!')

with sess.as_default():
    # Tensorboard writer
    writer = tf.summary.FileWriter(hp.logdir, sess.graph)
    for i in range(hp.num_train_epochs):
        indexs = shuffle_one(arr)
        for j in range(num_batchs - 1):
            i1 = indexs[j * hp.batch_size:min((j + 1) * hp.batch_size, num_train_samples)]

            # Get features
            input_id_ = select(input_ids, i1)
            input_mask_ = select(input_masks, i1)
            segment_id_ = select(segment_ids, i1)
            label_id_ = normalization_label(select(label_ids, i1))

            # Feed dict
            fd = {MODEL.input_ids: input_id_,
                  MODEL.input_masks: input_mask_,
                  MODEL.segment_ids: segment_id_,
                  MODEL.label_ids: label_id_}

            # Optimizer
            sess.run(MODEL.optimizer, feed_dict=fd)

            # Tensorboard
            if j % hp.summary_step == 0:
                summary, glolal_step = sess.run([MODEL.merged, MODEL.global_step], feed_dict=fd)
                writer.add_summary(summary, glolal_step)

            # Save Model
            if j % (num_batchs // hp.num_saved_per_epoch) == 0:
                if not os.path.exists(os.path.join(pwd, hp.file_model_save)):
                    os.makedirs(os.path.join(pwd, hp.file_model_save))
                saver.save(sess, os.path.join(pwd, hp.file_model_save, 'model' + '_%s_%s.ckpt' % (str(i), str(j))))

            # Log
            if j % hp.print_step == 0:
                fd = {MODEL.input_ids: input_id_,
                      MODEL.input_masks: input_mask_,
                      MODEL.segment_ids: segment_id_,
                      MODEL.label_ids: label_id_}
                loss = sess.run(MODEL.loss, feed_dict=fd)
                print('Time:%s, Epoch:%s, Batch number:%s/%s, Loss:%s' % (
                time_now_string(), str(i), str(j), str(num_batchs), str(loss)))
    print('Train finished')
