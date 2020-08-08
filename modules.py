# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:01:45 2019

@author: cm
"""


import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
from classifier_multi_label_textcnn.hyperparameters import Hyperparamters as hp



def cell_lstm(inputs,hidden_size,is_training):
    """
    inputs shape: (batch_size,sequence_length,embedding_size)
    hidden_size: rnn hidden size
    """
    with tf.variable_scope('cell_lstm'):
  
        cell_forward = tf.contrib.rnn.BasicLSTMCell(hidden_size/2)
        cell_backward = tf.contrib.rnn.BasicLSTMCell(hidden_size/2)
        cell_forward = DropoutWrapper(cell_forward, 
                                      input_keep_prob=1.0, 
                                      output_keep_prob=0.5 if is_training else 1)
        cell_backward = DropoutWrapper(cell_backward, 
                                       input_keep_prob=1.0, 
                                       output_keep_prob=0.5 if is_training else 1)                
        
        print('cell_forward: ',cell_forward )
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_forward,
                                                          cell_backward,
                                                          inputs,
                                                          dtype=tf.float32)
        forward_out, backward_out = outputs
        outputs = tf.concat([forward_out, backward_out], axis=2)
        # 激活函数
        outputs = tf.nn.leaky_relu(outputs, alpha=0.2)            
        value = tf.transpose(outputs, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0] - 1))            
        return last#(?,768)



def cell_cnn(inputs,is_training):
    sequence_length = hp.sequence_length
    conv1_left = hp.conv1_left
    num_filters = hp.num_filters
    #with tf.variable_scope('cell_cnn'):

    # 卷积
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    # 最后一个维度增加：-1
    inputs2 = tf.expand_dims(inputs, -1)
            
    # 第一层卷积
    with tf.variable_scope("conv1"):
        W_conv1 = tf.truncated_normal([conv1_left, 5 , 1, num_filters], stddev=0.1,name='W1')
        b_conv1 = tf.constant(0.1, shape=[num_filters],name='b1')
        h_conv1 = tf.nn.relu(conv2d(inputs2, W_conv1) + b_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1, 
                                 ksize=[1, sequence_length - conv1_left + 1, 1, 1], 
                                 strides=[1, 1, 1, 1], 
                                 padding='VALID')
        print ('W_conv1:',W_conv1.shape)# (100, 5, 1, 384)
        print ('h_conv1:',h_conv1.shape)#(16, 101, 764, 384)
        print ('h_pool1:',h_pool1.shape)#(16, 1, 764, 384)

        
    # 第二层卷积
    with tf.variable_scope("conv2"):
        W_conv2 = tf.truncated_normal([1, 1, num_filters, num_filters * 2], stddev=0.1,name='W2')
        b_conv2 = tf.constant(0.1, shape=[num_filters * 2],name='b2')
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')

        print ('W_conv2:',W_conv2.shape)#(1, 1, 384, 768)
        print ('b_conv2:',b_conv2.shape)#(768,)
        print ('h_conv2:',h_conv2.shape)#(16, 1, 764, 768)
        print ('h_pool2:',h_pool2.shape)#(16, 1, 382, 768)
        
    #维度获取
    L = h_pool2.get_shape().as_list()
    s = L[1] * L[2] * L[3]
    h_pool3_flat = tf.reshape(h_pool2, [-1, s])#.h_pool2_flat: (?, 768/4*64)   

    # 为了减少过拟合，我们在输出层之前加入dropout。
    outputs = tf.nn.dropout(h_pool3_flat, keep_prob=0.5 if is_training else 1)    
    return outputs   #(4, 24576)

def cell_lstm_cnn(inputs,hidden_size,is_training):
    """
    将CNN和BiLSTM的输出拼接在一起
    """
    # Output BiLSTM
    output_lstm = cell_lstm(inputs,hidden_size,is_training)
    # Output CNN
    output_cnn = cell_cnn(inputs,is_training)
    # 拼接output_lstm和output_cnn
    output = tf.concat((output_lstm, output_cnn), -1)
    return output        



def cell_textcnn(inputs,is_training):
    # 最后一个维度增加：-1
    inputs_expand = tf.expand_dims(inputs, -1)
    # Create a convolution + maxpool layer for each filter size
    pooled_outputs = []
    with tf.name_scope("TextCNN"):
        for i, filter_size in enumerate(hp.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, hp.embedding_size, 1, hp.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),dtype=tf.float32, name="W")
                b = tf.Variable(tf.constant(0.1, shape=[hp.num_filters]),dtype=tf.float32, name="b")
                conv = tf.nn.conv2d(
                                    inputs_expand,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                                        h,
                                        ksize=[1, hp.sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name="pool")
                pooled_outputs.append(pooled)
    # Combine all the pooled features
    num_filters_total = hp.num_filters * len(hp.filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    # Dropout
    h_pool_flat_dropout = tf.nn.dropout(h_pool_flat, keep_prob=hp.keep_prob if is_training else 1)
    return h_pool_flat_dropout
            

