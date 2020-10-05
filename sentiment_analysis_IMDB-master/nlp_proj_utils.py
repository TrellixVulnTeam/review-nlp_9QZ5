# -*- coding: utf-8 -*-
import os
import tarfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wget

from typing import NamedTuple, Tuple, Dict
from pandas import DataFrame
from functools import partial

DataMeta = NamedTuple('DataMeta', [
    ('name', str), ('url', str)])

_IMDB = DataMeta(
    name='imdb',
    url='https://www.dropbox.com/s/l9pj9hy2ans3phi/imdb.tar.gz?dl=1')

_GLOVE = DataMeta(
    name='glove',
    url='https://www.dropbox.com/s/g5pkso42wq2ipti/glove.tar.gz?dl=1'
)

def _download_data_if_needed(datas_meta: DataMeta) -> str:
    """
    Download and extract dataset if needed
    return the path to the dataset
    """
    path = os.path.join('resources', datas_meta.name)
    zip_path = path + '.tar.gz'

    if os.path.exists(path):
        print('data already available, skip downloading.')
    else:
        print('start downloading...')
        wget.download(datas_meta.url, zip_path)

        print('start extracting compressed files...')
        with tarfile.open(zip_path) as tar:
            tar.extractall('resources')
        os.remove(zip_path)

        print('data files are now available at %s' % path)
    return path

def _get_train_test_df(data_meta: DataMeta) -> Tuple[DataFrame, DataFrame]:
    path = _download_data_if_needed(data_meta)
    train, test = tuple(
        pd.read_csv(os.path.join(path, file))
        for file in ['train.csv', 'test.csv'])
    print('{} loaded successfully.'.format(data_meta.name))
    return train, test

def sentences_to_indices(text, word_to_index, max_len, topwords):
    """
    Return a array of indices of a given sentence.
    The sentence will be trimed/padded to max_len, only the word in topwords and glove will be left.

    Args:
        X (np.ndarray): Input array of sentences, the shape is (m,)  where m is the number of sentences, each sentence is a str. 
        Example X: array(['Sentence 1', 'Setence 2'])
        word_to_index (dict[str->int]): map from a word to its index in vocabulary

    Return:
        indices (np.ndarray): the shape is (m, max_len) where m is the number of sentences
    """
    X = np.asarray(text)
    m = X.shape[0]
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        sentence_words = X[i].lower().split()
        j = 0
        for w in sentence_words:
            if(w in topwords and word_to_index.get(w)!= None):
                X_indices[i, j] = word_to_index[w]
                j += 1
                if j >= max_len:
                    break
            
    return X_indices

def load_glove_vecs() -> Tuple[Dict[str, int], Dict[str, np.array]]:
    """
    Download (if necessary) and read GloVe. Two mappings are returned.
    1. Word to index, mapping from word to its index in vocabulary,
       needed for building Embedding layer in Keras)
    2. Word to vector, mapping from word to its vec
    """
    path = _download_data_if_needed(_GLOVE)
    path = os.path.join(path, 'glove.6B.50d.txt')
    print('loading glove... this may take a while...')
    with open(path, encoding='utf-8') as f:
        words = set()
        word_to_vec = {}
        for line in f:
            line_components = line.strip().split()
            curr_word = line_components[0]
            words.add(curr_word)
            word_to_vec[curr_word] = np.array(
                line_components[1:], dtype=np.float64)
        i = 1
        word_to_index = {}
        for w in sorted(words):
            word_to_index[w] = i
            i = i + 1
    print('glove loaded successfully.')
    return word_to_index, word_to_vec

def plot_history(his, metrics):
    """
    Given a history object returned from `fit` and the name of metrics,
    plot the curve of metrics against number of epochs.
    """
    for metric in metrics:
        plt.plot(his.history[metric], label=metric)
    plt.legend()

get_imdb_dataset = partial(_get_train_test_df, _IMDB)
