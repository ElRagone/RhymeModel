import numpy as np  # Numpy is used to have more control over the datatypes.
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


class Ngram:

    def __init__(self):
        self.hdhm = {}  # 2d hash map for saving doc ngram pairs.
        self.docs = set()  # set for saving which docs exists.
        self.ngrams = set()  # set for saving which ngram hashes exists.
        self.minhashes = {}  # Saves the minhashes per document

    def ngram(self, document, w=6, token='char'):
        buf = ['' for i in range(w)]
        with open(document, 'r') as doc:
            self.docs |= {document}
            if token is 'char':
                while True:
                    c = doc.read(1)
                    if not c:
                        break
                    elif c is '\n':
                        continue
                    else:
                        for i in range(w-1):
                            buf[i] = buf[i+1]
                        buf[w-1] = c
                        if buf[0] is not '':
                            ngram = ''
                            for t in buf:
                                ngram += t
                            ngramhash = hash(ngram) % (2**32)
                            ngram32 = np.uint32(ngramhash)
                            self.ngrams = {ngram32}
                            self.hdhm[(document, ngram32)] = True
            elif token is 'word':
                word_buf = ''
                while True:
                    c = doc.read(1)
                    if not c:
                        break
                    elif c is not '\n' and c is not ' ' and c is not '\t':
                        word_buf += c
                    else:
                        for i in range(w-1):
                            buf[i] = buf[i+1]
                        buf[w-1] = word_buf
                        if buf[0] is not '':
                            ngram = ''
                            for t in buf:
                                ngram += t
                                ngram += ' '
                            ngramhash = hash(ngram) % (2**32)
                            ngram32 = np.uint32(ngramhash)
                            self.ngrams |= {ngram32}
                            self.hdhm[(document, ngram32)] = True
                        word_buf = ''

            else:
                print('invalid token')
                exit(0)

    '''
    Jaccard function for determining similarity between sets.
    '''
    def jaccard(self, set1, set2):
        intersect = np.intersect1d(set1, set2)
        union = np.union1d(set1, set2)
        return float(len(intersect))/float(len(union))

    def cosine(self, a, b):
        return dot(a, b)/(norm(a)*norm(b))


def build_artist_vectors():
    grammer = Ngram()
    print 'Building the map'
    for f in tqdm(os.listdir('songs/')):
        grammer.ngram('songs/'+f)
    print 'Done building the map'
    lst = [(art.split('---')[0][6:], hsh) for art, hsh in grammer.hdhm.keys()]
    dic = {}
    print 'building a dict representation'
    for key, value in tqdm(lst):
        try:
            dic[key].append(value)
        except KeyError:
            dic[key] = []
    print 'done having a dict representation'
    jac = pd.DataFrame(index=dic.keys(), columns=dic.keys())
    print 'building a distance matrix'

    possible_values = {hsh for _, hsh in lst}
    cos = pd.DataFrame(index=dic.keys(), columns=possible_values)
    cos = cos.fillna(0)
    for artist, ngram in tqdm(lst):
        cos = cos.set_value(artist, ngram, cos.loc[artist, ngram]+1)
    cosmatrix = pd.DataFrame(index=dic.keys(), columns=dic.keys())

    vecs = {}
    for i in range(len(dic.keys())):
        vecs[dic.keys()[i]] = cos.as_matrix()[i, :]

    for col in tqdm(dic.keys()):
        for row in dic.keys():
            cosscore = grammer.cosine(vecs[col], vecs[row])
            cosmatrix.set_value(col, row, cosscore)
    cosmatrix.round(2)
    cosmatrix.to_csv('cos_artist.csv')


def build_bagofwords():
    split = [[s[0], s[1][:-4]] for s in [f.split('---') for f in
             os.listdir('songs/')]]
    y = [s[0] for s in split]
    X = [s[1] for s in split]
    skf = StratifiedKFold(n_splits=10)
    iteration = 0
    for tr1, te1 in skf.split(X, y):
        print iteration
        train = np.array(split)[tr1]
        test = np.array(split)[te1]
        allG = Ngram()
        trainG = Ngram()
        testG = Ngram()
        print 'reading the files and make ngrams (2x)'
        for f in tqdm(train):
            allG.ngram('songs/'+f[0]+'---'+f[1]+'.txt')
            trainG.ngram('songs/'+f[0]+'---'+f[1]+'.txt')
        for f in tqdm(test):
            allG.ngram('songs/'+f[0]+'---'+f[1]+'.txt')
            testG.ngram('songs/'+f[0]+'---'+f[1]+'.txt')
        print 'splitting the artists'
        lst_all = [(art.split('---')[0][6:], hsh) for art, hsh
                   in allG.hdhm.keys()]
        lst_tr = [(art.split('---')[0][6:], hsh)
                  for art, hsh in trainG.hdhm.keys()]
        lst_te = [(art[6:], hsh) for art, hsh in testG.hdhm.keys()]

        dic_tr = {}
        print 'building a dict representation'
        for key, value in tqdm(lst_all):
            try:
                dic_tr[key].append(value)
            except KeyError:
                dic_tr[key] = []
        possible_values = {hsh for _, hsh in lst_all}

        cos_tr = pd.DataFrame(columns=dic_tr.keys(), index=possible_values)
        cos_tr = cos_tr.fillna(0)

        for artist, ngram in tqdm(lst_tr):
            cos_tr = cos_tr.set_value(ngram, artist,
                                      cos_tr.loc[ngram, artist]+1)

        cos_tr.to_csv('train_mat_split{}.csv'.format(iteration), index=False)
        cos_te = []
        dic_te = {}

        for key, value in tqdm(lst_te):
            try:
                dic_te[key].append(value)
            except KeyError:
                dic_te[key] = []

        cos_te = pd.DataFrame(columns=dic_te.keys(), index=possible_values)
        cos_te = cos_te.fillna(0)

        for artist, ngram in tqdm(lst_te):
            cos_te = cos_te.set_value(ngram, artist,
                                      cos_te.loc[ngram, artist]+1)

        cos_te.to_csv('test_mat_split{}.csv'.format(iteration), index=False)
        iteration += 1


def create_artist_cos_matrix():
    lst_all = [(art.split('---')[0][6:], hsh) for art, hsh in allG.hdhm.keys()]
    lst_tr = [(art.split('---')[0][6:], hsh)
              for art, hsh in trainG.hdhm.keys()]
    lst_te = [(art[6:], hsh) for art, hsh in testG.hdhm.keys()]
    dic_tr = {}
    print 'building a dict representation'
    for key, value in tqdm(lst_all):
        try:
            dic_tr[key].append(value)
        except KeyError:
            dic_tr[key] = []
    possible_values = {hsh for _, hsh in lst_all}

    cos_tr = pd.DataFrame(columns=dic_tr.keys(), index=possible_values)
    cos_tr = cos_tr.fillna(0)

    for artist, ngram in tqdm(lst_tr):
        cos_tr = cos_tr.set_value(ngram, artist, cos_tr.loc[ngram, artist]+1)

    dic_te = {}

    for key, value in tqdm(lst_te):
        try:
            dic_te[key].append(value)
        except KeyError:
            dic_te[key] = []

    cos_te = pd.DataFrame(columns=dic_te.keys(), index=possible_values)
    cos_te = cos_te.fillna(0)

    for artist, ngram in tqdm(lst_te):
        cos_te = cos_te.set_value(ngram, artist, cos_te.loc[ngram, artist]+1)

    cos_te.to_csv('test_mat.csv', index=False)
    cos_tr.to_csv('train_mat.csv', index=False)


def prediction():
    a = Ngram()
    for i in range(10):
        cos_tr = pd.read_csv('train_mat_split{}.csv'.format(i))
        cos_te = pd.read_csv('test_mat_split{}.csv'.format(i))

        cosmatrix = pd.DataFrame(index=list(cos_te), columns=list(cos_tr))
        for row in list(cos_te):
            for col in list(cos_tr):
                cvalue = a.cosine(cos_te[row], cos_tr[col])
                cosmatrix.set_value(row, col, cvalue)
        df = cosmatrix
        df.to_csv('cos{}.csv'.format(i))
        df = pd.read_csv('cos{}.csv'.format(i), header=None)
        df = df.as_matrix()
        output = np.array([[row[0].split('---')[0], row[1]] for row in df])
        true_l = output[:, 0]
        pred_l = output[:, 1]
        labels = list(set(true_l))
        print "split{}: micro f1:{}".format(i, f1_score(true_l, pred_l,
                                                        average='micro',
                                                        labels=labels))
