# -*- coding: utf-8 -*-
"""
Created on Thu May 30 20:44:42 2021

@author: cm
"""

import os
import tensorflow as tf
from classifier_multi_label_denses import modeling
from classifier_multi_label_denses import optimization
from classifier_multi_label_denses.utils import time_now_string
from classifier_multi_label_denses.hyperparameters import Hyperparamters as hp
from classifier_multi_label_denses.classifier_utils import ClassifyProcessor



num_labels = hp.num_labels
processor = ClassifyProcessor() 
bert_config_file = os.path.join(hp.bert_path,'albert_config.json')
bert_config = modeling.AlbertConfig.from_json_file(bert_config_file)



class NetworkAlbert(object):
    def __init__(self,is_training):
        # Training or not
        self.is_training = is_training    
        
        # Placeholder       
        self.input_ids = tf.placeholder(tf.int32, shape=[None, hp.sequence_length], name='input_ids')
        self.input_masks = tf.placeholder(tf.int32, shape=[None,  hp.sequence_length], name='input_masks')
        self.segment_ids = tf.placeholder(tf.int32, shape=[None,  hp.sequence_length], name='segment_ids')
        self.label_ids = tf.placeholder(tf.int32, shape=[None,hp.num_labels], name='label_ids')
               
        # Load BERT model
        self.model = modeling.AlbertModel(
                                    config=bert_config,
                                    is_training=self.is_training,
                                    input_ids=self.input_ids,
                                    input_mask=self.input_masks,
                                    token_type_ids=self.segment_ids,
                                    use_one_hot_embeddings=False)


        # Get the feature vector by BERT
        output_layer = self.model.get_pooled_output()            
        print('output_layer',output_layer)#(?, 384)
                
        # Hidden size 
        hidden_size = output_layer.shape[-1].value    
           
        with tf.name_scope("Full-connection"):  
            loss_num_label = []
            logits_num_label = []
            for i in range(hp.num_labels):
                output_weights = tf.get_variable(
            	              "output_weights%s"%str(i), [2, hidden_size],
            	              initializer=tf.truncated_normal_initializer(stddev=0.02))        
                output_bias = tf.get_variable(
            	              "output_bias%s"%str(i), [2], initializer=tf.zeros_initializer())# 
                logits = tf.matmul(output_layer, output_weights, transpose_b=True)
                logits = tf.nn.bias_add(logits, output_bias)
                logits_num_label.append(logits)
                one_hot_labels = tf.one_hot(self.label_ids[:,i], depth=2, dtype=tf.int32)
                per_example_loss = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels,logits=logits)
                loss_num_label.append(tf.reduce_mean(per_example_loss))
            self.logits_num_label = tf.transpose(tf.stack(logits_num_label, 0),[1,0,2])
            self.loss_num_label = tf.stack(loss_num_label, 0)
            self.probabilities = tf.nn.sigmoid(self.logits_num_label)


        with tf.variable_scope("Prediction"):             
            # Prediction               
            self.predictions = tf.to_int32(tf.argmax(self.probabilities,2)) 
   
        with tf.variable_scope("loss"):            
            # Summary for tensorboard
            if self.is_training:
	            self.accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.predictions, self.label_ids)))
	            tf.summary.scalar('accuracy', self.accuracy) 
                                               
            # Initial embedding by BERT
            ckpt = tf.train.get_checkpoint_state(hp.saved_model_path)
            checkpoint_suffix = ".index"
            if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + checkpoint_suffix):
                print('='*10,'Restoring model from checkpoint!','='*10)
                print("%s - Restoring model from checkpoint ~%s" % (time_now_string(),
                                                                    ckpt.model_checkpoint_path))
            else:                   
                print('='*10,'First time load BERT model!','='*10)
                tvars = tf.trainable_variables()
                if hp.init_checkpoint:
                   (assignment_map, initialized_variable_names) = \
                     modeling.get_assignment_map_from_checkpoint(tvars,
                                                                 hp.init_checkpoint)
                   tf.train.init_from_checkpoint(hp.init_checkpoint, assignment_map)
                                
            # Loss and Optimizer
            if self.is_training:
                # Global_step
                self.global_step = tf.Variable(0, name='global_step', trainable=False)                   
                self.loss = tf.reduce_mean(self.loss_num_label)

                # Optimizer BERT
                train_examples = processor.get_train_examples(hp.data_dir)
                num_train_steps = int(
                    len(train_examples) / hp.batch_size * hp.num_train_epochs)
                num_warmup_steps = int(num_train_steps * hp.warmup_proportion)
                print('num_train_steps',num_train_steps)
                self.optimizer = optimization.create_optimizer(self.loss,
                                                                hp.learning_rate, 
                                                                num_train_steps, 
                                                                num_warmup_steps,
                                                                hp.use_tpu,
                                                                Global_step=self.global_step)    

                # Summary for tensorboard                 
                tf.summary.scalar('loss', self.loss)
                self.merged = tf.summary.merge_all()
                
                
                
                
if __name__ == '__main__':
    # Load model
    albert = NetworkAlbert(is_training=True)










