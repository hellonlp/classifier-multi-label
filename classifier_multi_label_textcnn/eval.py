# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 10:56:43 2021

@author: cm
"""


import pandas as pd
from classifier_multi_label_textcnn.utils import load_csv,load_excel,save_csv,shuffle_two,save_txt
from classifier_multi_label_textcnn.predict import get_label_multi
from classifier_multi_label_textcnn.utils import cut_list


if __name__ == '__main__':
    print("参考下面的方法")
    ## 参考 https://github.com/hellonlp/classifier-multi-label/blob/master/evaluation/eval_multi_label.py
