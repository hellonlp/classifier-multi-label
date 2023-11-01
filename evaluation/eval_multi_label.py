# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 15:26:11 2023

@author: Chen Ming
"""


class EvalMultiLabel(object):
   """ 评估多标签的精确率、召回率和F1值 """
  
    def __init__(self, y_trues, y_preds):     
        self.y_trues = y_trues
        self.y_preds = y_preds
    
        def some_samples(y_trues,y_preds):
            """ 评估多个样本的TP、FN、FP、TN """ 
            if len(y_trues) == len(y_preds):
                tp = 0
                fn = 0
                fp = 0
                tn = 0
                for i in range(len(y_trues)):
                    y_true = y_trues[i]
                    y_pred = y_preds[i]
                    tpi,fni,fpi,tni = single_sample(y_true,y_pred)
                    tp = tp + tpi
                    fn = fn + fni
                    fp = fp + fpi
                    tn = tn + tni
                return tp,fn,fp,tn
            else:
                print('Different length between y_trues and y_preds!')
                return 0,0,0,0
              
        def single_sample(y_true,y_pred):
            """ 评估单个样本的TP、FN、FP、TN """        
            y_true = list(set(y_true))
            y_pred = list(set(y_pred))
            y_ = list(set(y_true) | set(y_pred))
            K = len(y_)
            tp1 = 0
            fn1 = 0
            fp1 = 0
            tn1 = 0
            for i in range(len(y_)):
                if y_[i] in y_true and y_[i] in y_pred:
                    tp1 = tp1 + 1/K
                elif y_[i] in y_true and y_[i] not in y_pred:
                    fn1 = fn1 + 1/K
                elif y_[i] not in y_true and y_[i] in y_pred:
                    fp1 = fp1 + 1/K  
                elif y_[i] not in y_true and y_[i] not in y_pred:
                    tn1 = tn1 + 1/K
            return tp1,fn1,fp1,tn1
        
        self.tp,self.fn,self.fp,self.tn = some_samples(self.y_trues,self.y_preds)
        self.recall = self.tp/(self.tp+self.fn)
        self.precision = self.tp/(self.tp+self.fp)
        self.f1 = 2*self.recall*self.precision/(self.precision+self.recall)
        
        
if __name__ == '__main__':
    y_trues = [['a','b','c'],['a','f','c','h']] #真实标签
    y_preds = [['a','d'],['a','d']] #预测标签
    EML = EvalMultiLabel(y_trues,y_preds)  
    print(EML.precision)
    print(EML.recall)
    print(EML.f1)
  
