# 简介
1、本项目是在tensorflow版本1.15.0的基础上做的训练和测试。  
2、本项目为中文的多标签文本分类。  
3、欢迎大家联系 www.hellonlp.com  
4、albert_small_zh_google对应的百度云下载地址：  
   链接：https://pan.baidu.com/s/1RKzGJTazlZ7y12YRbAWvyA  
   提取码：wuxw  

 
# 使用方法
1、准备数据  
数据格式为：classifier_multi_label_textcnn/data/test_onehot.csv  
2、参数设置  
参考脚本 hyperparameters.py，直接修改里面的数值即可。  
3、训练  
```
python train.py
```
4、预测
```
python predict.py
```
注意：推理时需要把model/save中的模型复制到model/load中，并修改model/load中的checkpoint文件的内容为当前模型名称，例如：model_checkpoint_path: "model_xx_xx.ckpt"。
。


 
 # 知乎解读
https://zhuanlan.zhihu.com/p/158622992
