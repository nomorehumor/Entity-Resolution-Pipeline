from utils import get_words_ngrams
from collections import defaultdict
from nltk.tokenize import word_tokenize
import pandas as pd
import itertools
import numpy as np

def create_ngram_word_blocks(df, column, n):
    blocks = {}
    for idx, row in df.iterrows():
        ngrams_list = get_words_ngrams(row[column], n)
        for ngram in ngrams_list:
            if ngram not in blocks:
                blocks[ngram] = []
            blocks[ngram].append(idx) # only append idx of row in dataframe to save more space
    return blocks

def get_candidate_pairs_between_blocks(blocks1, blocks2):
    """
    Helper function to create candidate pairs between blocks
    """
    candidate_pairs = set()
    for ngram in blocks1:
        if ngram in blocks2:
            pairs = [(id1, id2) for id1 in blocks1[ngram] for id2 in blocks2[ngram]]
            candidate_pairs.update(pairs) # add all candidate pairs to set
    return candidate_pairs

def token_blocking(df1, df2, stop_words: set):
    blocks1 = defaultdict(list)
    blocks2 = defaultdict(list)

    for idx, row in enumerate(df1.itertuples()):

        string = " ".join([str(value) for value in row if not pd.isna(value)])
        tokens = set(
            [word for word in word_tokenize(string) if word not in stop_words]
        )

        for token in tokens:
            blocks1[token].append(idx)

    for idx, row in enumerate(df2.itertuples()):

        string = " ".join([str(value) for value in row if not pd.isna(value)])
        tokens = set(
            [word for word in word_tokenize(string) if word not in stop_words]
        )

        for token in tokens:
            blocks2[token].append(idx)

    blocks1 = {
        key: indices
        for key, indices in blocks1.items()
        if len(indices) < 1000 and len(indices) > 1
    }
    blocks2 = {
        key: indices
        for key, indices in blocks2.items()
        if len(indices) < 1000 and len(indices) > 1
    }

    pairs = [list(pair) for key in (blocks1.keys() & blocks2.keys()) for pair in
             list(itertools.product(blocks1[key], blocks2[key]))]

    return np.array(pairs)

def blocking_by_year(df1, df2, cols=['year_acm', 'year_dblp']):
    b1 = defaultdict(list)
    b2 = defaultdict(list)

    for idx, key in df1[cols[0]].items():
        if key:
            b1[key].append(idx)

    for idx, key in df2[cols[1]].items():
        if key:
            b2[key].append(idx)

    pairs = [list(pair) for key in b1.keys() for pair in list(itertools.product(b1[key], b2[key]))]

    return np.array(pairs)