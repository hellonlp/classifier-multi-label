# Text Classification Multi-Label
[![Python](https://img.shields.io/badge/python-3.7.6-blue?logo=python&logoColor=FED643)](https://www.python.org/downloads/release/python-376/)
[![Pytorch](https://img.shields.io/badge/tensorflow-1.15.0-red?logo=tensorflow)](https://www.tensorflow.org/versions/)


## 一、简介
<img src="https://github.com/hellonlp/classifier-multi-label/blob/master/imgs/01.png" width="700" height="320">
1. 多分类任务中一条数据只有一个标签，但这个标签可能有多种类别。比如判定某个人的性别，只能归类为"男性"、"女性"其中一个。再比如判断一个文本的情感只能归类为"正面"、"中面"或者"负面"其中一个。  
2. 多标签分类任务中一条数据可能有多个标签，每个标签可能有两个或者多个类别。例如，一篇新闻可能同时归类为"娱乐"和"运动"，也可能只属于"娱乐"或者其它类别。

## 二、算法

**4种实现方法**
```
├── classifier_multi_label
    └── classifier_multi_label
    └── classifier_multi_label_textcnn
    └── classifier_multi_label_denses
    └── classifier_multi_label_seq2seq
```

### 第1种
classifier_multi_label
- 使用BERT第一个token[CLS]的向量，维度为(batch_size,hidden_size)。
- 链接：https://zhuanlan.zhihu.com/p/164873441


### 第2种
classifier_multi_label_textcnn
- 使用BERT输出的三维向量，维度为(batch_size,sequence_length,hidden_size)，然后做为输入进入TextCNN层。
- 链接：https://zhuanlan.zhihu.com/p/158622992


### 第3种
classifier_multi_label_denses
- 使用BERT第一个token[CLS]的向量，维度为(batch_size,hidden_size)，然后通过多个二分类(全连接层)来解决多标签分类问题。
- 链接：https://zhuanlan.zhihu.com/p/263573628


### 第4种
classifier_multi_label_seq2seq
- 使用BERT输出的三维向量，维度为(batch_size,sequence_length,hidden_size)，然后做为输入进入seq2seq+attention层。
- 链接：https://zhuanlan.zhihu.com/p/260743336


## 对比文章
- 链接：https://zhuanlan.zhihu.com/p/152140983  

