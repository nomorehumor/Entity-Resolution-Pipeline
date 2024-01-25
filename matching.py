import pandas as pd
from data_loading import get_vector_datasets
from utils import get_symbol_ngrams
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import numpy as np


def matching(df1, df2, df_blocking, matching, weights=None):
    if matching == 'cosine':
        vector_matching(df1, df2, df_blocking)
    else:
        if weights == None:
            raise ValueError("Weights must be initialized if using string matching")
        string_matching(df_blocking, sim=matching, weights=weights)


def baseline_matching(df1, df2, match, weights):
    all_pairs = np.array([list(pair) for pair in itertools.product(df1.index, df2.index)])
    bs_df = pd.concat([df1.loc[all_pairs[:, 0]].reset_index(),
                       df2.loc[all_pairs[:, 1]].reset_index()], axis=1)
    string_matching(bs_df, sim=match, weights=weights)
    return bs_df


def string_matching(df, sim='jaccard', weights=[0.33, 0.33, 0.33]):
    if sim == 'jaccard':
        df['similarity'] = df.apply(lambda x: weights[0] * jaccard_sim(x.title_acm, x.title_dblp)
                                    + weights[1] * jaccard_sim(x.authors_acm, x.authors_dblp)
                                    + weights[2] * int(x.year_acm == x.year_dblp)
                                    , axis=1)
    elif sim == 'trigram':
        df['similarity'] = df.apply(
            lambda x: weights[0] * trigram_sim(get_symbol_ngrams(x.title_acm), get_symbol_ngrams(x.title_dblp))
            + weights[1] * trigram_sim(get_symbol_ngrams(x.authors_acm), get_symbol_ngrams(x.authors_dblp))
            + weights[2] * int(x.year_acm == x.year_dblp)
            , axis=1)
    elif sim == 'levenshtein':
        df['similarity'] = df.apply(lambda x: weights[0] * levenshtein_sim(x.title_acm, x.title_dblp)
                                    + weights[1] * levenshtein_sim(x.authors_acm, x.authors_dblp)
                                    + weights[2] * int(x.year_acm == x.year_dblp)
                                    , axis=1)


def vector_matching(df1, df2, df_blocking):
    # Vectorization using TF-IDF
    vector_space1, vector_space2 = get_vector_datasets(df1, df2)

    pairs = df_blocking[["index_acm", "index_dblp"]]
    for i, idx1, idx2 in pairs.itertuples():
        similarity_func = lambda x: cosine_similarity(
            vector_space1[x["index_dblp"]].reshape(1, -1),
            vector_space2[x["index_acm"]].reshape(1, -1)
        )[0, 0]
        df_blocking["similarity"] = df_blocking.apply(similarity_func, axis=1)


def jaccard_sim(s1, s2):
    s_intersection = set(set(s1.split()).intersection(set(s2.split())))
    s_union = set(s1.split()).union(set(s2.split()))
    return len(s_intersection) / len(s_union)


def trigram_sim(ngram_1, ngram_2):
    return 2 * len(ngram_1.intersection(ngram_2)) / (len(ngram_1) + len(ngram_2))


def levenshtein_dist(s1, s2):
    if len(s1) == 0 or len(s2) == 0:
        return max(len(s1), len(s2))  # returns the length of the non-empty string
    if s1[0] == s2[0]:
        return levenshtein_dist(s1[1:], s2[1:])
    else:
        return 1 + min(
            levenshtein_dist(s1[1:], s2),  # deletion
            levenshtein_dist(s1, s2[1:]),  # insertion
            levenshtein_dist(s1[1:], s2[1:])  # substitution
        )


def levenshtein_sim(s1, s2):
    return 1 - levenshtein_dist(s1, s2) / max(len(s1), len(s2))
