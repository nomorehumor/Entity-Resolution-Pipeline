import itertools
import re
from collections import defaultdict
import numpy as np
import pandas as pd


def data_preparation():
    df1 = pd.read_csv('ACM_1995_2004.csv', sep='|')
    df2 = pd.read_csv('DBLP_1995_2004.csv', sep='|')
    df = pd.concat([df1, df2], ignore_index=True)
    df['Title'] = column_normalization(df['Title'])
    df['Authors'] = column_normalization(df['Authors'])
    df.dropna(ignore_index=True, inplace=True)

    return df


def blocking(values, return_pairs=False):
    buckets = defaultdict(list)
    for idx, key in enumerate(values):
        if key:
            buckets[key].append(idx)

    if return_pairs:
        buckets_w_pairs = {}
        for key in buckets.keys():
            buckets_w_pairs[key] = [
                pair for pair in list(itertools.combinations(buckets[key], 2))
            ]

        return buckets_w_pairs

    return buckets


def matching(df, sim='jaccard', threshold = 0.8):
    if sim == 'jaccard':
        df['similarity'] = df.apply(lambda x: jaccard_sim(x.title_one, x.title_two), axis=1)
    elif sim == 'trigram':
        df['similarity'] = df.apply(lambda x: trigram_sim(x.title_one, x.title_two), axis=1)
    elif sim == 'levenshtein':
        df['similarity'] = df.apply(lambda x: levenshtein_sim(x.title_one, x.title_two), axis=1)
    
    matching_pairs = df[df['similarity'] > threshold]
    return matching_pairs

def column_normalization(col_values):
    return (col_values
            .str.lower()
            .replace("[^a-z0-9]", " ", regex=True)
            .replace(" +", " ", regex=True)
            .str.strip())


def jaccard_sim(s1, s2):
    s_intersection = set(s1.split()).intersection(s2.split())
    s_union = set(s1.split()).union(s2.split())
    return len(s_intersection) / len(s_union)


def trigram_sim(s1, s2):
    ngram_1 = get_trigrams(s1)
    ngram_2 = get_trigrams(s2)
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

def get_trigrams(text, number=3):
    if not text:
        return set()
    words = [f' {x} ' for x in re.split(r'\W+', text.lower()) if x.strip()]
    words[0] = f' {words[0]}'
    words[-1] = f'{words[-1]} '
    ngrams = set()
    for word in words:
        for x in range(0, len(word) - number + 1):
            ngrams.add(word[x:x + number])
    return ngrams


def entity_resolution():
    df = data_preparation()
    blocks_on_year = blocking(df['Year'], return_pairs=True)
    pairs = np.array([x for (k, v) in blocks_on_year.items() for x in v])
    pairs_df = pd.concat([df['Title'].loc[pairs[:, 0]].rename('title_one').reset_index(),
                          df['Title'].loc[pairs[:, 1]].rename('title_two').reset_index()], axis=1)
    df_matches = matching(pairs_df, sim='jaccard')
    return df_matches
    