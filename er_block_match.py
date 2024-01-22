import itertools
import re
import string
import time

import nltk
from nltk.corpus import stopwords
from collections import defaultdict
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize


def data_preparation():
    df1 = pd.read_csv('ACM_1995_2004.csv', sep='|')
    df2 = pd.read_csv('DBLP_1995_2004.csv', sep='|')
    df1.dropna(ignore_index=True, inplace=True)
    df2.dropna(ignore_index=True, inplace=True)
    preprocessing(df1)
    df1.index.names = ['index_acm']
    df1.columns = ['paperID_acm', 'title_acm', 'authors_acm', 'venue_acm', 'year_acm']
    preprocessing(df2)
    df2.columns = ['paperID_dblp', 'title_dblp', 'authors_dblp', 'venue_dblp', 'year_dblp']
    df2.index.names = ['index_dblp']

    return df1, df2


def preprocessing(df):
    df['Title'] = (df['Title'].str.lower()
                   .replace("[^a-z0-9]", " ", regex=True)
                   .replace(" +", " ", regex=True)
                   .str.strip())
    df['Authors'] = (df['Authors']
                     .str.lower()
                     .replace("[^a-z0-9]", " ", regex=True)
                     .replace(" +", " ", regex=True)
                     .str.strip())

    df['Venue'] = (df['Venue']
                   .str.lower()
                   .replace(" +", " ", regex=True)
                   .str.strip())


def blocking(df1, df2, cols=['year_acm', 'year_dblp']):
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


def matching(df, sim='jaccard', threshold = 0.8):
    if sim == 'jaccard':
        df['similarity'] = df.apply(lambda x: jaccard_sim(x.title_one, x.title_two), axis=1)
    elif sim == 'trigram':
        df['similarity'] = df.apply(lambda x: trigram_sim(x.title_one, x.title_two), axis=1)
    elif sim == 'levenshtein':
        df['similarity'] = df.apply(lambda x: levenshtein_sim(x.title_one, x.title_two), axis=1)
    
    matching_pairs = df[df['similarity'] > threshold]
    return matching_pairs
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


def matching(df, sim='jaccard', weights=[0.33, 0.33, 0.33]):
    if sim == 'jaccard':
        df['jaccard_sim'] = df.apply(lambda x: weights[0] * jaccard_sim(x.title_acm, x.title_dblp)
                                               + weights[1] * jaccard_sim(x.authors_acm, x.authors_dblp)
                                               + weights[2] * int(x.year_acm == x.year_dblp)
                                     , axis=1)
    elif sim == 'trigram':
        df['trigram_sim'] = df.apply(
            lambda x: weights[0] * trigram_sim(get_ngrams(x.title_acm), get_ngrams(x.title_dblp))
                      + weights[1] * trigram_sim(get_ngrams(x.authors_acm), get_ngrams(x.authors_dblp))
                      + weights[2] * int(x.year_acm == x.year_dblp)
            , axis=1)

def jaccard_sim(s1, s2):
    s_intersection = set(set(s1.split()).intersection(set(s2.split())))
    s_union = set(s1.split()).union(set(s2.split()))
    return len(s_intersection) / len(s_union)


def trigram_sim(ngram_1, ngram_2):
    return 2 * len(ngram_1.intersection(ngram_2)) / (len(ngram_1) + len(ngram_2))

def levenshtein_dist(s1, s2):
    if len(s1) == 0 or len(s2) == 0:
        return max(len(s1), len(s2)) # returns the length of the non-empty string
    if s1[0] == s2[0]:
        return levenshtein_dist(s1[1:], s2[1:])
    else:
        return 1 + min(
            levenshtein_dist(s1[1:], s2), # deletion
            levenshtein_dist(s1, s2[1:]), # insertion
            levenshtein_dist(s1[1:], s2[1:]) # substitution
        )

def levenshtein_sim(s1,s2):
    return 1 - levenshtein_dist(s1,s2) / max(len(s1), len(s2))

def get_ngrams(text, number=3):
    if not text:
        return set()
    text = ' ' * (number - 1) + text + ' ' * (number - 1)
    ngrams = set()
    for x in range(0, len(text) - number + 1):
        ngrams.add(text[x:x + number])
    return ngrams


def blocking_and_matching(df1, df2, block='st', match='jacc',  weights=[0.3, 0.3, 0.3]):

    if block == 'token':
        stop_words = set(stopwords.words('english') + list(string.punctuation))
        blocks = token_blocking(df1[['title_acm', 'authors_acm']], df2[['title_dblp', 'authors_dblp']], stop_words)
        blocks = np.unique(blocks, axis=0)
    else:
        blocks = blocking(df1, df2)

    df_blocking = pd.concat([df1.loc[blocks[:, 0]].reset_index(),
                             df2.loc[blocks[:, 1]].reset_index()], axis=1)

    matching(df_blocking, sim=match, weights=weights)

    return df_blocking


def baseline_matching(df1, df2, match, weights):
    all_pairs = np.array([list(pair) for pair in itertools.product(df1.index, df2.index)])
    bs_df = pd.concat([df1.loc[all_pairs[:, 0]].reset_index(),
                              df2.loc[all_pairs[:, 1]].reset_index()], axis=1)
    matching(bs_df, sim=match, weights=weights)
    return bs_df


def evaluate(df, bs_df, f, match='jaccard'):
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    for threshold in thresholds:
        if match=='trigram':
            bs_df_trigram = bs_df[bs_df.trigram_sim > threshold].sort_values(by=['trigram_sim'],
                                                                             ascending=False)
            match_df_trigram = df[df.trigram_sim > threshold].sort_values(by=['trigram_sim'],
                                                                          ascending=False)
            f1, prec, rec = f1_evaluation(match_df_trigram, bs_df_trigram)
            print(f'Trigram: threshold: {threshold}, f1: {f1}, precision: {prec}, recall: {rec}', file=f)
        else:
            bs_df_jaccard = bs_df[bs_df.jaccard_sim > threshold].sort_values(by=['jaccard_sim'],
                                                                             ascending=False)
            match_df_jaccard = df[df.jaccard_sim > threshold].sort_values(by=['jaccard_sim'],
                                                                          ascending=False)
            f1, prec, rec = f1_evaluation(match_df_jaccard, bs_df_jaccard)
            print(f'Jaccard: threshold: {threshold}, f1: {f1}, precision: {prec}, recall: {rec}', file=f)


def f1_evaluation(df, bs_df):
    tp = len(pd.merge(df, bs_df, how='inner', on=['index_acm', 'index_dblp']))
    fp = len(df) - tp
    fn = len(bs_df) - tp
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall


def entity_resolution_experiments(save_csv=False):
    df1, df2 = data_preparation()
    matches = ['jaccard', 'trigram', 'jaccard', 'jaccard', 'trigram', 'jaccard']
    blocks = ['st', 'st', 'st', 'token', 'token', 'token']
    weights = [[0.7, 0.3, 0], [0.7, 0.3, 0], [0.8, 0.2, 0], [0.5, 0.2, 0.3], [0.5, 0.2, 0.3], [0.5, 0.3, 0.2]]
    with open("output.txt", "a") as f:
        for i in range(len(matches)):
            print(f'{blocks[i]}, {matches[i]}, {weights[i]}', file=f)
            start= time.time()
            df = blocking_and_matching(df1, df2, block=blocks[i], match=matches[i], weights=weights[i])
            print(f'Time needed for blocking and matchign: {time.time() - start}', file=f)

            start = time.time()
            bs_df = baseline_matching(df1, df2, match=matches[i], weights=weights[i])
            print(f'Time needed for baseline creation: {time.time() - start}', file=f)

            evaluate(df, bs_df, f, match=matches[i])

            if save_csv:
                w = f'{weights[i][0]*10}{weights[i][1]*10}{weights[2][0]*10}'
                df.to_csv(f'{matches[i]}_{blocks[i]}_{w}.csv')


if __name__ == "__main__":
    entity_resolution_experiments()
