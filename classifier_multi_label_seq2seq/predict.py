# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:12:37 2019

@author: cm
"""


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
pwd = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tensorflow as tf
from classifier_multi_label_seq2seq.networks import NetworkAlbertSeq2Seq
from classifier_multi_label_seq2seq.hyperparameters import Hyperparamters as hp
from classifier_multi_label_seq2seq.classifier_utils import get_feature_test


class ModelAlbertSeq2seq(object, ):
    """
    Load Network Albert Seq2seq model
    """

    def __init__(self):
        self.albert, self.sess = self.load_model()

    @staticmethod
    def load_model():
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                albert = NetworkAlbertSeq2Seq(is_training=False)
                saver = tf.train.Saver()
                sess.run(tf.global_variables_initializer())
                checkpoint_dir = os.path.abspath(os.path.join(pwd, hp.file_model_load))
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                saver.restore(sess, ckpt.model_checkpoint_path)
        return albert, sess


MODEL = ModelAlbertSeq2seq()
print('Load model finished!')


def get_label(sentence):
    """
    Prediction of the sentence's sentiment.
    """

    feature = get_feature_test(sentence)
    fd = {MODEL.albert.input_ids: [feature[0]],
          MODEL.albert.input_masks: [feature[1]],
          MODEL.albert.segment_ids: [feature[2]],
          }
    output = MODEL.sess.run(MODEL.albert.predictions, feed_dict=fd)
    return sorted([hp.dict_id2label[i] for i in output[0][:, 0] if i != 1])


def get_labels(sentences):
    """
    Prediction of some sentences's sentiment.
    """
    features = [get_feature_test(str(sentence)) for sentence in sentences]
    fd = {MODEL.albert.input_ids: [feature[0] for feature in features],
          MODEL.albert.input_masks: [feature[1] for feature in features],
          MODEL.albert.segment_ids: [feature[2] for feature in features]}
    outputs = MODEL.sess.run(MODEL.albert.predictions, feed_dict=fd)
    return [sorted([hp.dict_id2label[i] for i in output[:, 0] if i != 1]) for output in outputs]


if __name__ == '__main__':
    # Test
    sentences = ['重量大小出差也很OK',
                 '轻薄本的通病',
                 'logo周围的槽打磨不平整',
                 '简约外观']
    for sentence in sentences:
        print(get_label(sentence))
