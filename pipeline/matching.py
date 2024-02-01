import time
import Levenshtein
import pandas as pd
from pipeline.data_loading import get_vector_datasets
from utils import create_cartesian_product_baseline, get_symbol_ngrams
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import numpy as np


def matching(df_acm, df_dblp, pairs, matching, weights=None):
    """Returns a dataframe with the index_acm, index_dblp and similarity score"""
    if matching == 'cosine':
        return vector_matching(df_acm, df_dblp, pairs)
    else:
        if weights == None:
            raise ValueError("Weights must be initialized if using string matching")
        return string_matching(df_acm, df_dblp, pairs, sim=matching, weights=weights)


def baseline_matching(df_acm, df_dblp, matching_function, weights):
    start = time.time()
    bs_pairs = np.array([list(pair) for pair in itertools.product(df_acm.index, df_dblp.index)])
    end_bs = time.time()
    print("Time needed for baseline creation: ", end_bs - start)

    matched_bs_df = matching(df_acm, df_dblp, bs_pairs, matching=matching_function, weights=weights)
    end_matching = time.time()
    print("Time needed for baseline matching: ", end_matching - end_bs)
    return matched_bs_df


def string_matching(df_acm, df_dblp, pairs, sim='jaccard', weights=[0.33, 0.33, 0.33]):
    # df = pd.DataFrame(columns=["index_acm", "index_dblp", "similarity"])
    # df['index_acm'] = pairs[:, 0]
    # df['index_dblp'] = pairs[:, 1]
    # df = df.merge(df_acm, left_on='index_acm', right_index=True)
    # df = df.merge(df_dblp, left_on='index_dblp', right_index=True)

    acm_title, dblp_title = 'title_acm', 'title_dblp'
    acm_authors, dblp_authors = 'authors_acm', 'authors_dblp'

    if sim == 'jaccard':
        sim_func = jaccard_sim
    elif sim == 'trigram':
        df_acm['trigram_title_acm'] = np.vectorize(get_symbol_ngrams)(df_acm['title_acm'])
        df_acm['trigram_authors_acm'] = np.vectorize(get_symbol_ngrams)(df_acm['authors_acm'])
        df_dblp['trigram_title_dblp'] = np.vectorize(get_symbol_ngrams)(df_dblp['title_dblp'])
        df_dblp['trigram_authors_dblp'] = np.vectorize(get_symbol_ngrams)(df_dblp['authors_dblp'])

        acm_title, dblp_title = 'trigram_title_acm', 'trigram_title_dblp'
        acm_authors, dblp_authors = 'trigram_authors_acm', 'trigram_authors_dblp'
        sim_func = trigram_sim
    elif sim == 'levenshtein':
        sim_func = levenshtein_sim


    df = pd.concat([df_acm.loc[pairs[:, 0]].reset_index(),
                    df_dblp.loc[pairs[:, 1]].reset_index()], axis=1)

    title_sim = np.vectorize(sim_func)(df[acm_title], df[dblp_title])
    authors_sim = np.vectorize(sim_func)(df[acm_authors], df[dblp_authors])
    year_sim = (df['year_acm'] == df['year_dblp']).astype(int)

    df['similarity'] = weights[0] * title_sim + weights[1] * authors_sim + weights[2] * year_sim

    return df


def vector_matching(df_acm, df_dblp, pairs):
    # Vectorization using TF-IDF
    vector_space1, vector_space2 = get_vector_datasets(df_acm, df_dblp)
    cosine_sim = cosine_similarity(vector_space1, vector_space2)

    df = pd.DataFrame(columns=["index_acm", "index_dblp", "similarity"])
    df['index_acm'] = pairs[:, 0]
    df['index_dblp'] = pairs[:, 1]
    df['similarity'] = cosine_sim[pairs[:, 0], pairs[:, 1]]
    return df


def jaccard_sim(s1, s2):
    s_intersection = set(set(s1.split()).intersection(set(s2.split())))
    s_union = set(s1.split()).union(set(s2.split()))
    return len(s_intersection) / len(s_union) if len(s_union) > 0 else 0


def trigram_sim(ngram_1, ngram_2):
    return 2 * len(ngram_1.intersection(ngram_2)) / (len(ngram_1) + len(ngram_2)) if len(ngram_1) + len(
        ngram_2) > 0 else 0


def levenshtein_sim(s1, s2):
    return 1 - Levenshtein.distance(s1, s2) / max(len(s1), len(s2)) if max(len(s1), len(s2)) > 0 else 0
