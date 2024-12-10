# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 20:03:12 2020

@author: cm
"""

import os
import tensorflow as tf

from classifier_multi_label_seq2seq import modeling
from classifier_multi_label_seq2seq import optimization
from classifier_multi_label_seq2seq.modules import encoder, decoder
from classifier_multi_label_seq2seq.hyperparameters import Hyperparamters as hp
from classifier_multi_label_seq2seq.utils import time_now_string
from classifier_multi_label_seq2seq.classifier_utils import ClassifyProcessor


num_labels = hp.num_labels
processor = ClassifyProcessor()
bert_config_file = os.path.join(hp.bert_path, 'albert_config.json')
bert_config = modeling.AlbertConfig.from_json_file(bert_config_file)


class NetworkAlbertSeq2Seq(object):
    def __init__(self, is_training):
        # Training or not
        self.is_training = is_training

        # Placeholder  
        self.input_ids = tf.placeholder(tf.int32, shape=[None, hp.sequence_length], name='input_ids')
        self.input_masks = tf.placeholder(tf.int32, shape=[None, hp.sequence_length], name='input_masks')
        self.segment_ids = tf.placeholder(tf.int32, shape=[None, hp.sequence_length], name='segment_ids')
        self.label_ids = tf.placeholder(tf.int32, shape=[None, None], name='label_ids')

        # Load BERT token features
        self.model = modeling.AlbertModel(
            config=bert_config,
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.input_masks,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)

        # Get tensor BERT
        self.output_layer_initial = self.model.get_sequence_output()

        # Hidden_size
        self.hidden_size = self.output_layer_initial.shape[-1].value

        with tf.name_scope("Encoder"):
            # Get input length of encoder 
            self.input_ids_sequence_length = tf.count_nonzero(self.output_layer_initial, axis=2, dtype=tf.int32)
            self.input_ids_length = tf.count_nonzero(self.input_ids_sequence_length, axis=1, dtype=tf.int32)

            # Encoder
            self.memory, self.encode_state = encoder(self.output_layer_initial,
                                                     self.hidden_size,
                                                     self.input_ids_length,
                                                     _is_training=self.is_training)
        with tf.name_scope("Decoder"):
            # Decoder
            self.outputs, self.alignments, self.mask, self.final_sequence_length = decoder(self.label_ids,
                                                                                           self.memory,
                                                                                           self.encode_state,
                                                                                           _is_training=is_training)

        # Initial embedding by BERT
        ckpt = tf.train.get_checkpoint_state(hp.saved_model_path)
        checkpoint_suffix = ".index"
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + checkpoint_suffix):
            print('=' * 10, 'Restoring model from checkpoint!', '=' * 10)
            print("%s - Restoring model from checkpoint ~%s" % (time_now_string(),
                                                                ckpt.model_checkpoint_path))
        else:
            tvars = tf.trainable_variables()
            if hp.init_checkpoint:
                (assignment_map, initialized_variable_names) = \
                    modeling.get_assignment_map_from_checkpoint(tvars,
                                                                hp.init_checkpoint)
                tf.train.init_from_checkpoint(hp.init_checkpoint, assignment_map)

        # Loss and Optimizer
        if self.is_training:
            with tf.name_scope("loss"):
                # Global step
                self.global_step = tf.Variable(0, name='global_step', trainable=False)

                # Prediction
                self.predictions = self.outputs.sample_id

                # Loss                
                per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ids,
                                                                                  logits=self.outputs.rnn_output)

                self.istarget = tf.to_float(tf.not_equal(self.label_ids, 0))
                self.loss = tf.reduce_sum(per_example_loss * self.istarget) / (tf.reduce_sum(self.istarget) + 1e-7)

                # Accuracy                
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.label_ids), tf.float32))

                # Summary for tensorboard  
                tf.summary.scalar('accuracy', self.accuracy)
                tf.summary.scalar('loss', self.loss)
                self.merged = tf.summary.merge_all()

            # Optimizer BERT
            num_train_examples = len(processor.get_train_examples(hp.data_dir))
            num_train_steps = int(
                num_train_examples / hp.batch_size * hp.num_train_epochs)
            num_warmup_steps = int(num_train_steps * hp.warmup_proportion)
            self.optimizer = optimization.create_optimizer(self.loss,
                                                           hp.learning_rate,
                                                           num_train_steps,
                                                           num_warmup_steps,
                                                           hp.use_tpu,
                                                           Global_step=self.global_step)

        else:
            if hp.is_beam_search:
                self.predictions = self.outputs.predicted_ids
                self.predictions_prob = self.outputs.beam_search_decoder_output[0]
                self.predictions_all = self.outputs.beam_search_decoder_output
            else:
                self.predictions = tf.expand_dims(self.outputs.sample_id, -1)
                self.probs = tf.expand_dims(self.outputs.rnn_output, -1)


if __name__ == '__main__':
    #  Load model
    albert = NetworkAlbertSeq2Seq(is_training=True)
