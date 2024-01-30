# Text Classification Multi-Label: 多标签文本分类
[![Python](https://img.shields.io/badge/python-3.7.6-blue?logo=python&logoColor=FED643)](https://www.python.org/downloads/release/python-376/)
[![Pytorch](https://img.shields.io/badge/tensorflow-1.15.0-red?logo=tensorflow)](https://www.tensorflow.org/versions/)     

<img src="https://github.com/hellonlp/classifier-multi-label/blob/master/imgs/HELLONLP.png" width="800">

<br/>

## 一、简介
### 1. 多元分类
   多分类任务中一条数据只有一个标签，但这个标签可能有多种类别。比如判定某个人的性别，只能归类为"男性"、"女性"其中一个。再比如判断一个文本的情感只能归类为"正面"、"中面"或者"负面"其中一个。
### 2. 多标签分类
   多标签分类任务中一条数据可能有多个标签，每个标签可能有两个或者多个类别。例如，一篇新闻可能同时归类为"娱乐"和"运动"，也可能只属于"娱乐"或者其它类别。    

<br/>

<img src="https://github.com/hellonlp/classifier-multi-label/blob/master/imgs/01b.png" width="800">

<br/>


## 二、算法

**4种实现方法**
```
├── classifier_multi_label
    └── classifier_multi_label
    └── classifier_multi_label_textcnn
    └── classifier_multi_label_denses
    └── classifier_multi_label_seq2seq
```

### 1. classifier_multi_label
<img src="https://github.com/hellonlp/classifier-multi-label/blob/master/imgs/base.png" width="700">  

- 使用BERT第一个token[CLS]的向量，维度为(batch_size,hidden_size)。  
- 使用了tf.nn.sigmoid_cross_entropy_with_logits的损失函数。
- 使用了tf.where函数来选择概率小于0.5的对应id。  

### 2. classifier_multi_label_textcnn
<img src="https://github.com/hellonlp/classifier-multi-label/blob/master/imgs/textcnn.png" width="700">  

- 使用BERT输出的三维向量，维度为(batch_size,sequence_length,hidden_size)，然后做为输入进入TextCNN层。
- 使用了tf.nn.sigmoid_cross_entropy_with_logits的损失函数。
- 使用了tf.where函数来选择概率小于0.5的对应id。   

### 3. classifier_multi_label_denses
<img src="https://github.com/hellonlp/classifier-multi-label/blob/master/imgs/denses01.png" width="700">  
<img src="https://github.com/hellonlp/classifier-multi-label/blob/master/imgs/denses02.png" width="700">  

- 使用BERT第一个token[CLS]的向量，维度为(batch_size,hidden_size)，然后通过多个二分类(全连接层)来解决多标签分类问题。
- 使用了tf.nn.softmax_cross_entropy_with_logits的损失函数。
- 使用了tf.argmax函数来选择输出最高概率。  

### 4. classifier_multi_label_seq2seq
<img src="https://github.com/hellonlp/classifier-multi-label/blob/master/imgs/seq2seq.png" width="700">  

- 使用BERT输出的三维向量，维度为(batch_size,sequence_length,hidden_size)，然后做为输入进入seq2seq+attention层。  
- 使用了tf.nn.softmax_cross_entropy_with_logits的损失函数。
- 使用了beam search 来解码输出概率。  
  
<br/>

## 三、实验
### 1. 训练过程
<img src="https://github.com/hellonlp/classifier-multi-label/blob/master/imgs/10.png" width="700">

### 2. 实验结果
<img src="https://github.com/hellonlp/classifier-multi-label/blob/master/imgs/09b.jpg" width="700">


### 3. 实验结论
- 如果对推理速度的要求不是非常高，基于ALBERT+Seq2Seq_Attention框架的多标签文本分类效果最好。   
- 如果对推理速度和模型效果要求都非常高，基于ALBERT+TextCNN会是一个不错的选择。  

<br/>

## 参考
[多标签文本分类介绍，以及对比训练](https://zhuanlan.zhihu.com/p/152140983)  
[多标签文本分类 [ALBERT]](https://zhuanlan.zhihu.com/p/164873441)  
[多标签文本分类 [ALBERT+TextCNN]](https://zhuanlan.zhihu.com/p/158622992)  
[多标签文本分类 [ALBERT+Multi_Denses]](https://zhuanlan.zhihu.com/p/263573628)  
[多标签文本分类 [ALBERT+Seq2Seq+Attention]](https://zhuanlan.zhihu.com/p/260743336)      




