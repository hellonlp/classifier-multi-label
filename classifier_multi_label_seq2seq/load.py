# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 20:00:59 2020

@author: cm
"""


from classifier_multi_label_seq2seq.hyperparameters import Hyperparamters as hp


def label2onehot(string):
    string = '|' if string == '' else string
    string_list = list(str(string).split('/')) + ['E']
    return [int(hp.dict_label2id.get(l)) for l in string_list]


def normalization_label(label_ids):
    max_length = max([len(l) for l in label_ids])
    return [l + [0] * (max_length - len(l)) if len(l) < max_length else l for i, l in enumerate(label_ids)]


if __name__ == '__main__':
    # Test
    label_ids = [[1, 2, 3], [1, 2], [2, 3, 4, 5, 6]]
    print(normalization_label(label_ids))
    #
#    from classifier_multi_label_seq2seq.utils import load_csv,save_csv
#    f = 'classifier_multi_label_seq2seq/data/test.csv'
#    df = load_csv(f)
#    contents = df['content'].tolist()
#    labels = df['label'].tolist()
#    ls = ['产品整体评价','机身颜色','外观设计','重量尺寸','机身材质','外壳做工']
#    labels_new = []
#    for l in labels:
#        l1 = str(l).split('/')
#        l1_new = []
#        for li in l1:
#            if li in ls:
#                l1_new.append(li)
#        labels_new.append(l1_new)
#    #
#    import pandas as pd
#    df = pd.DataFrame(columns=['content','label'])
#    df['content'] = contents
#    df['label'] = ['/'.join(l) for l in labels_new]
#    file_csv = f = 'classifier_multi_label_seq2seq/data/test0.csv'
#    save_csv(df,file_csv,encoding='utf-8-sig')
