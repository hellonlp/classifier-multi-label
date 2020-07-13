# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:23:12 2018

@author: cm
"""



import os
import sys
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pwd)
from classification_multi_label_01.utils import load_vocabulary



class Hyperparamters:
    # Parameters    
    print_step = 100
    summary_step = 10
    save_model_step = 300
    train_rate = 0.999   
    batch_size = 64          
    batch_size_predict = 1
    learning_rate = 5e-5 
    
    # Load dict
    dict_id2label,dict_label2id = load_vocabulary(os.path.join(pwd,'data','vocabulary_label.txt') )
    label_vocabulary = list(dict_id2label.values())

    # Optimization parameters
    num_train_epochs = 20
    warmup_proportion = 0.1
    use_tpu = None
    do_lower_case = True
    num_labels = len(list(dict_id2label))
    num_filters = 128

    # CNN parameters
    sequence_length = 60 
    conv1_left = 100
    
    # TextCNN parameters
    filter_sizes = [2,3,4,5,6,7]
    embedding_size = 384
    keep_prob = 0.5
    
    
    # BERT model
    model = 'albert_small_zh_google'
    bert_path = os.path.join(pwd,model)
    data_dir = os.path.join(pwd,'data')
    vocab_file = os.path.join(pwd,model,'vocab_chinese.txt')
    init_checkpoint = os.path.join(pwd,model,'albert_model.ckpt')
    saved_model_path = os.path.join(pwd,'model')    
    
    
    
    
    
    
    
    


    
    