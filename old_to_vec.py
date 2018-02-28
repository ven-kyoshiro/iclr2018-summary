# _*_ coding:utf-8 _*_
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pickle
import umap
from sklearn.datasets import load_digits
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import time
import numpy as np
from old_tech import old_tech

'''
I  used code from here
https://qiita.com/asian373asian/items/1be1bec7f2297b8326cf

and also
https://qiita.com/cheerfularge/items/27a55ebde4a671880666

matplotlib error
https://qiita.com/Kodaira_/items/1a3b801c7a5a41c9ce49
''' 

def prepro(file_name):  # read pickle and save as txt
    # papers =[[id,title,abstruct,html],]
    with open(file_name, mode='rb') as f:
        papers = pickle.load(f)
    text = ''
    for pp in papers:
        tx = pp[1]+' '
        tx += pp[2].replace('\n',' ')
        text = text + '\n' + tx
    text = text[1:]
    with open('train_data.txt', 'w') as f:
        f.write(text)

def paper2vec():
    prepro('iclr_2018_papers.pickle')
    f = open('train_data.txt','r')
    trainings = [TaggedDocument(
                words = data.split(),tags = [i]) for i,data in enumerate(f)]    
    # TODO:parametor tuning
    m = Doc2Vec(
        documents= trainings, 
        dm = 1, 
        vector_size=300,
        window=15,
        min_count=1,
        workers=2)
    m.save("model/doc2vec.model")

def coloring(file_name):
# papers =[[id,title,abstruct,html],]
    with open(file_name, mode='rb') as f:
        papers = pickle.load(f)
    col = []
    for p in papers:
        if 'reinforce' in p[1] or 'Reinforce' in p[1]:
            col.append('red')
        elif 'GAN' in p[1]:
            col.append('blue')
        elif 'immitation' in p[1] or 'Immitation' in p[1]:
            col.append('orange')
        elif 'LSTM' in p[1]:
            col.append('green')
        else:
            col.append('black')
    return col

def visualize(m, col):
    OLD = True
    fs = 20,20
    # digits = load_digits()
    # digits.target = [float(digits.target[i]) for i in range(len(digits.target))]
    if OLD:
        src = np.array(old_tech())
    else:
        src = np.array([m.docvecs[i] for i in range(len(m.docvecs))])
    print(src)
    #UMAP5
    start_time = time.time()
    embedding = umap.UMAP().fit_transform(src)
    interval = time.time() - start_time
    
    fig, ax = plt.subplots(figsize=(fs))
    ax.scatter(embedding[:,0],embedding[:,1],color = col)
    for i, e in enumerate(embedding):
        ax.annotate(str(i+1), xy=(e[0],e[1]), size=8)
    #plt.colorbar()
    plt.savefig('old_umap.png')

    #raw
    if len(src[0]) == 2:
        plt.clf()
        fig, ax = plt.subplots(figsize=(fs))
        ax.scatter(src[:,0],src[:,1],color = col)
        for i, s in enumerate(src):
            ax.annotate(str(i+1), xy=(s[0],s[1]), size=8)
            plt.scatter(src[:,0],src[:,1],color = col)
        plt.savefig('raw.png')

    # t-SNE
    plt.clf()
    start_time2 = time.time()
    tsne_model = TSNE(n_components=2)
    tsne = tsne_model.fit_transform(src)
    interval2 = time.time() - start_time2
    fig, ax = plt.subplots(figsize=(fs))
    ax.scatter(tsne[:,0],tsne[:,1],color = col)
    for i, t in enumerate(tsne):
        ax.annotate(str(i+1), xy=(t[0],t[1]), size=8)
    
    plt.scatter(tsne[:,0],tsne[:,1], color = col)
    #plt.scatter(tsne[:,0],tsne[:,1],c=digits.target,cmap=cm.tab10)
    #plt.colorbar()
    plt.savefig('old_tsne.png')
    print('model_vecs:' + str(len(m.docvecs)))
    print('color:'+str(len(col)))
    print('src:'+str(len(src)))
    print('umap : {}s'.format(interval))
    print('tsne : {}s'.format(interval2))


def main():
    paper2vec()
    m = Doc2Vec.load("model/doc2vec.model")
    print(m.docvecs[1])# これでベクトルが取り出せる？
    col = coloring('iclr_2018_papers.pickle')
    visualize(m,col)

if __name__ == '__main__':
    main()
