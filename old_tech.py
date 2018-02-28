# -*- codig:utf-8 -*-
from gensim import corpora, matutils
'''
This code is from https://qiita.com/yasunori/items/31a23eb259482e4824e2
https://qiita.com/yasunori/items/31a23eb259482e4824e2
'''


def old_tech():
    f = open('train_data.txt')
    data1 = f.read()
    f.close()
    data1 = data1.replace('(', '')
    data1 = data1.replace(')', '')
    data1 = data1.lower()
    data = data1.split('\n')
    data = [d.split(' ') for d in data]

    dictionary = corpora.Dictionary(data)
    dictionary.filter_extremes(no_below=3, no_above=0.3)
    dictionary.save_as_text('old_tech_dict.txt')
    # dictionary = corpora.Dictionary.load_from_text('livedoordic.txt')
    # dict としてみる　dictionary.token2id
    vecs = [list(matutils.corpus2dense(
                [dictionary.doc2bow(d)],
                num_terms=len(dictionary)).T[0]) for d in data]
    return vecs
