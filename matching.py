import pandas as pd
from data_loading import get_vector_datasets
from utils import get_symbol_ngrams
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import numpy as np

def matching(df1, df2, df_block_pairs, matching, weights=None):
    if matching == 'cosine':
        return vector_matching(df1, df2, df_block_pairs)
    else:
        if weights == None:
            raise ValueError("Weights must be initialized if using string matching")
        return string_matching(df_block_pairs, sim=matching, weights=weights)
        
def baseline_matching(df1, df2, match, weights):
    all_pairs = np.array([list(pair) for pair in itertools.product(df1.index, df2.index)])
    bs_df = pd.concat([df1.loc[all_pairs[:, 0]].reset_index(),
                              df2.loc[all_pairs[:, 1]].reset_index()], axis=1)

    matched_bs_df = matching(df1, df2, bs_df, matching=match, weights=weights)
    return matched_bs_df

def string_matching(df, sim='jaccard', weights=[0.33, 0.33, 0.33]):
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
    return df[["index_acm", "index_dblp", "similarity"]]
def vector_matching(df_acm, df_dblp, df_block_pairs):
    # Vectorization using TF-IDF
    vector_space1, vector_space2 = get_vector_datasets(df_acm, df_dblp)
    cosine_sim = cosine_similarity(vector_space1, vector_space2)

    # for i in range ck_pairs.loc[i, "similarity"] = cosine_sim[i, j]
    # pairs = df_block_pairs[["index_acm", "index_dblp"]]
    # for i, (index_acm, index_dblp) in pairs.iterrows():
    #     df_block_pairs.loc[i, "similarity"] = cosine_similarity(
    #         vector_space1[index_acm].reshape(1, -1), 
    #         vector_space2[index_dblp].reshape(1, -1)
    #     )[0, 0]
    print(len(df_block_pairs))
    matching_pairs = []
    for id_acm in range(len(cosine_sim)):
        for id_dblp in range(len(cosine_sim[id_acm])):
            # matched_df["similarity"].loc[len(matched_df)] = [id_acm, id_dblp, cosine_sim[id_acm,id_dblp]]
            matching_pairs.append((id_acm,id_dblp,cosine_sim[id_acm,id_dblp]))
            
    matched_df = pd.DataFrame(matching_pairs, columns=["index_acm", "index_dblp", "similarity"])
    return matched_df
    
    # similarity_func = lambda x: cosine_similarity(
    #     vector_space1[x["index_acm"]].reshape(1, -1), 
    #     vector_space2[x["index_dblp"]].reshape(1, -1)
    # )[0, 0]
    # df_block_pairs["similarity"] = df_block_pairs.apply(similarity_func, axis=1)       

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