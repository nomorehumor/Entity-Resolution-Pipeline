import time
import pandas as pd
from data_loading import get_vector_datasets
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
    df = pd.concat([df_acm.loc[pairs[:, 0]].reset_index(),
                    df_dblp.loc[pairs[:, 1]].reset_index()], axis=1)
    
    if sim == 'jaccard':
        df['similarity'] = df.apply(lambda x: weights[0] * jaccard_sim(x.title_acm, x.title_dblp)
                        + weights[1] * jaccard_sim(x.authors_acm, x.authors_dblp)
                        + weights[2] * int(x.year_acm == x.year_dblp)
                                    , axis=1)
    elif sim == 'trigram':
        df['similarity'] = df.apply(lambda x: weights[0] * trigram_sim(get_symbol_ngrams(x.title_acm), get_symbol_ngrams(x.title_dblp))
                      + weights[1] * trigram_sim(get_symbol_ngrams(x.authors_acm), get_symbol_ngrams(x.authors_dblp))
                      + weights[2] * int(x.year_acm == x.year_dblp)
                                    , axis=1)
    elif sim == 'levenshtein':
        df['similarity'] = df.apply(lambda x: weights[0] * levenshtein_sim(x.title_one, x.title_two)
                                    + weights[1] * levenshtein_sim(x.authors_acm, x.authors_dblp), axis=1)
    return df

def vector_matching(df_acm, df_dblp, pairs):
    # Vectorization using TF-IDF
    vector_space1, vector_space2 = get_vector_datasets(df_acm, df_dblp)
    cosine_sim = cosine_similarity(vector_space1, vector_space2)
    
    df = pd.DataFrame(columns=["index_acm", "index_dblp", "similarity"])
    df['index_acm'] = pairs[:, 0]
    df['index_dblp'] = pairs[:, 1]
    df['similarity']= cosine_sim[pairs[:, 0], pairs[:, 1]]
    return df

def jaccard_sim(s1, s2):
    s_intersection = set(set(s1.split()).intersection(set(s2.split())))
    s_union = set(s1.split()).union(set(s2.split()))
    return len(s_intersection) / len(s_union) if len(s_union) > 0 else 0


def trigram_sim(ngram_1, ngram_2):
    return 2 * len(ngram_1.intersection(ngram_2)) / (len(ngram_1) + len(ngram_2)) if len(ngram_1) + len(ngram_2) > 0 else 0

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
    return 1 - levenshtein_dist(s1,s2) / max(len(s1), len(s2)) if max(len(s1), len(s2)) > 0 else 0