from pipeline.utils import get_candidate_pairs_between_blocks, get_words_ngrams, get_token_blocks
from collections import defaultdict
import itertools
import numpy as np
from nltk.corpus import stopwords
import string


def blocking(df_acm, df_dblp, blocking_scheme, params):
    if blocking_scheme == "ngram_word_blocks":
        blocks1 = create_ngram_word_blocks(df_acm, "Combined_acm",  params["n"])
        blocks2 = create_ngram_word_blocks(df_dblp, "Combined_dblp", params["n"])
        candidate_pairs_set = get_candidate_pairs_between_blocks(blocks1, blocks2)
        return candidate_pairs_set
    elif blocking_scheme == 'token':
        stop_words = set(stopwords.words('english') + list(string.punctuation))
        blocks = token_blocking(df_acm[['title_acm', 'authors_acm']], df_dblp[['title_dblp', 'authors_dblp']], stop_words)
        blocks = np.unique(blocks, axis=0)
        return blocks
    elif blocking_scheme == 'st':
        blocks = blocking_by_year(df_acm, df_dblp)
        return blocks
        

def create_ngram_word_blocks(df, column, n):
    blocks = {}
    for idx, row in df.iterrows():
        ngrams_list = get_words_ngrams(row[column], n)
        for ngram in ngrams_list:
            if ngram not in blocks:
                blocks[ngram] = []
            blocks[ngram].append(idx) # only append idx of row in dataframe to save more space
    return blocks


def token_blocking(df1, df2, stop_words: set):
    blocks1 = get_token_blocks(df1, stop_words)
    blocks2 = get_token_blocks(df2, stop_words)

    pairs = [list(pair) for key in (blocks1.keys() & blocks2.keys()) for pair in
             list(itertools.product(blocks1[key], blocks2[key]))]

    return np.array(pairs)


def blocking_by_year(df_acm, df_dblp):
    b1 = defaultdict(list)
    b2 = defaultdict(list)

    for idx, key in df_acm['year_acm'].items():
        if key:
            b1[key].append(idx)

    for idx, key in df_dblp['year_dblp'].items():
        if key:
            b2[key].append(idx)

    pairs = [list(pair) for key in b1.keys() for pair in list(itertools.product(b1[key], b2[key]))]

    return np.array(pairs)
